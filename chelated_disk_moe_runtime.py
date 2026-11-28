"""CPU disk-MoE expert paging and greedy speculative verification for PV7 Bundle 56."""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import random
import statistics
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from chelated_pv7_spine.cpu_io_cage import ONE_GIB_BYTES

MIB = 1024 ** 2
MEMORY_MAX = 8 * ONE_GIB_BYTES
RUNGS = (
    "R1_mmap_random",
    "R2_direct_pread_lru",
    "R3_extent_coalesced",
    "R4_speculative_draft_prefetch",
    "R5_iq2_reap_compressed_separate_quality",
    "R6_chelated_cylinder_ordered_pinned_moe",
)
LOSSLESS_RUNGS = (1, 2, 3, 4, 6)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidentBudget:
    target_weights: int
    draft_weights: int
    kv_cache: int
    page_tables: int
    expert_lru: int
    expert_hot_pin: int
    read_buffers: int
    inference_workspace: int
    routing_and_runtime: int
    kernel_and_page_cache: int

    def __post_init__(self):
        if any(type(getattr(self, field.name)) is not int or getattr(self, field.name) < 0 for field in fields(self)):
            raise ValueError("Each memory charge must be a nonnegative byte count")
        if self.total > MEMORY_MAX:
            raise ValueError(f"Resident memory budget exceeded: {self.total} > {MEMORY_MAX}")

    @property
    def total(self):
        return sum(asdict(self).values())


@dataclass(frozen=True)
class Extent:
    offset: int
    size: int


def positional_read(fd, size, offset):
    """Use pread on POSIX and the offset-bearing ReadFile API on Windows."""
    if hasattr(os, "pread"):
        return os.pread(fd, size, offset)

    import msvcrt
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = [
            ("internal", ctypes.c_size_t), ("internal_high", ctypes.c_size_t),
            ("offset", wintypes.DWORD), ("offset_high", wintypes.DWORD), ("event", wintypes.HANDLE),
        ]

    read_file = ctypes.WinDLL("kernel32", use_last_error=True).ReadFile
    read_file.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(Overlapped)]
    read_file.restype = wintypes.BOOL
    buf = ctypes.create_string_buffer(size)
    count = wintypes.DWORD()
    position = Overlapped(offset=offset & 0xFFFFFFFF, offset_high=offset >> 32)
    if not read_file(msvcrt.get_osfhandle(fd), buf, size, ctypes.byref(count), ctypes.byref(position)):
        error = ctypes.get_last_error()
        if error != 38:  # ERROR_HANDLE_EOF
            raise ctypes.WinError(error)
    return buf.raw[:count.value]


class ExpertPager:
    """Bound decoded expert input by byte capacity and issue explicit positional reads."""

    alignment = 4096

    def __init__(
        self, path, extents, rung, *, cache_bytes=0, pin_bytes=0, max_read_bytes=8 * MIB,
        direct=False, physical_offset=None,
    ):
        if rung < 1 or rung > len(RUNGS) or min(cache_bytes, pin_bytes) < 0 or max_read_bytes <= 0:
            raise ValueError("Invalid rung or pager limits")
        if pin_bytes and rung != 6:
            raise ValueError("The hot-pin cache belongs to R6")
        self.extents = extents
        self.rung = rung
        self.cache_bytes = cache_bytes
        self.pin_bytes = pin_bytes
        self.max_read_bytes = max_read_bytes
        self.physical_offset = physical_offset
        self.direct = direct
        self.bytes_read = 0
        self.read_calls = 0
        self.demand = Counter()
        self.cache = OrderedDict()
        self.pinned = OrderedDict()
        self.cache_size = 0
        self.pinned_size = 0
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        self.mapping = None
        self.direct_fd = None
        try:
            size = os.fstat(self.fd).st_size
            if not extents or any(
                e.offset < 0 or not 0 < e.size <= max_read_bytes or e.offset + e.size > size
                for e in extents.values()
            ):
                raise ValueError("Expert extents exceed file or read buffer bounds")
            if rung == 1:
                self.mapping = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
            if direct:
                self.direct_fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
        if self.direct_fd is not None:
            os.close(self.direct_fd)
        os.close(self.fd)

    def reset(self, *, cold):
        self.bytes_read = 0
        self.read_calls = 0
        if cold:
            self.cache.clear()
            self.pinned.clear()
            self.cache_size = self.pinned_size = 0
            self.demand.clear()
            if hasattr(os, "posix_fadvise"):
                if self.mapping is not None:
                    self.mapping.madvise(mmap.MADV_DONTNEED)
                os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def _read(self, start, end):
        length = end - start
        if self.mapping is not None:
            data = self.mapping[start:end]
            transferred = length
        elif self.direct_fd is not None:
            aligned = start // self.alignment * self.alignment
            displacement = start - aligned
            nbytes = (displacement + length + self.alignment - 1) // self.alignment * self.alignment
            with mmap.mmap(-1, nbytes) as scratch:
                transferred = os.preadv(self.direct_fd, [scratch], aligned)
                data = scratch[displacement:min(transferred, displacement + length)]
        else:
            data = positional_read(self.fd, length, start)
            transferred = len(data)
        if len(data) != length:
            raise OSError(f"Short expert read at {start}: {len(data)} of {length} bytes")
        self.bytes_read += transferred
        self.read_calls += 1
        return data

    def _ranges(self, keys):
        if self.rung in (1, 2):
            return [(self.extents[key].offset, self.extents[key].offset + self.extents[key].size, [key])
                    for key in keys]
        # Group logically adjacent spans first, then order the reads by on-disk address on physical HDD.
        spans = []
        for key in sorted(keys, key=lambda item: self.extents[item].offset):
            extent = self.extents[key]
            end = extent.offset + extent.size
            if spans:
                start, previous_end, members = spans[-1]
                contiguous = (
                    self.physical_offset is None
                    or self.physical_offset(extent.offset) - self.physical_offset(start) == extent.offset - start
                )
                if contiguous and extent.offset <= previous_end + 32 and max(previous_end, end) - start <= self.max_read_bytes:
                    members.append(key)
                    spans[-1] = (start, max(previous_end, end), members)
                    continue
            spans.append((extent.offset, end, [key]))
        if self.rung == 6 and self.physical_offset is not None:
            spans.sort(key=lambda span: self.physical_offset(span[0]))
        return spans

    def _remember(self, key, data):
        if len(data) <= self.cache_bytes:
            while self.cache_size + len(data) > self.cache_bytes:
                _, discarded = self.cache.popitem(last=False)
                self.cache_size -= len(discarded)
            self.cache[key] = data
            self.cache_size += len(data)

    def _promote(self, key, data):
        if key in self.pinned or self.demand[key] < 2 or len(data) > self.pin_bytes:
            return
        while self.pinned_size + len(data) > self.pin_bytes:
            victim = min(self.pinned, key=self.demand.__getitem__)
            if self.demand[victim] >= self.demand[key]:
                return
            self.pinned_size -= len(self.pinned.pop(victim))
        if key in self.cache:
            self.cache_size -= len(self.cache.pop(key))
        self.pinned[key] = data
        self.pinned_size += len(data)

    def fetch(self, keys, *, prefetch=False):
        keys = list(dict.fromkeys(keys))
        if not prefetch:
            self.demand.update(keys)
        result = {}
        for key in keys:
            if key in self.pinned:
                result[key] = self.pinned[key]
            elif key in self.cache:
                self.cache.move_to_end(key)
                result[key] = self.cache[key]
        missing = [key for key in keys if key not in result]
        for start, end, members in self._ranges(missing):
            data = self._read(start, end)
            for key in members:
                extent = self.extents[key]
                payload = data[extent.offset - start:extent.offset - start + extent.size]
                self._remember(key, payload)
                if not prefetch:
                    result[key] = payload
        if not prefetch:
            for key, data in result.items():
                self._promote(key, data)
        return result if not prefetch else None


def greedy_decode(target, prompt, *, max_tokens, draft=None, lookahead=4, eos=()):
    """Verify draft tokens one at a time on the same target path used by plain decoding."""
    if not prompt or min(max_tokens, lookahead) < 1:
        raise ValueError("A positive token limit and a nonempty prompt are required")
    target.reset()
    if draft is not None:
        draft.reset()
    prefix = list(prompt)
    next_token = target.start(prefix)
    emitted = []
    proposed_count = accepted_count = 0
    while len(emitted) < max_tokens:
        width = min(lookahead, max_tokens - len(emitted))
        proposals = [] if draft is None else draft.propose(prefix, width)
        proposed_count += len(proposals)
        if proposals:
            target.prefetch(proposals)
        for index in range(max(1, len(proposals))):
            matches = index < len(proposals) and proposals[index] == next_token
            accepted_count += int(matches)
            emitted.append(next_token)
            prefix.append(next_token)
            if next_token in eos or len(emitted) == max_tokens:
                return emitted, proposed_count, accepted_count
            next_token = target.advance(next_token)
            if not matches:
                break
    return emitted, proposed_count, accepted_count


def select_dev_best(rows):
    """Freeze one Q4 comparator using pooled DEV time, without holdout or R5 rows."""
    def throughput(rung):
        cells = rows[rung]
        if not cells:
            raise ValueError(f"Missing DEV timing for R{rung}")
        return sum(row["tokens"] for row in cells) / sum(row["seconds"] for row in cells)

    return max(range(1, 5), key=lambda rung: (throughput(rung), -rung))


def compare_lossless(reference, baseline, candidate, *, seed=56, resamples=4000):
    """Bootstrap paired throughput by prompt, keeping cold/warm replicates together."""
    def indexed(rows):
        result = {(row["suite"], row["id"], row["trial"], row["cache"]): row for row in rows}
        if not result or len(result) != len(rows):
            raise ValueError("Missing or duplicated timing cell")
        return result

    oracle, control, treatment = map(indexed, (reference, baseline, candidate))
    if oracle.keys() != control.keys() or oracle.keys() != treatment.keys():
        raise ValueError("Unpaired holdout timing cells")
    equal = all(oracle[key]["token_sha256"] == control[key]["token_sha256"] == treatment[key]["token_sha256"]
                and oracle[key]["tokens"] == control[key]["tokens"] == treatment[key]["tokens"] for key in oracle)
    per_suite = defaultdict(lambda: defaultdict(list))
    for key, left in control.items():
        per_suite[key[0]][key[1]].append((left["seconds"], treatment[key]["seconds"]))
    rng = random.Random(seed)
    strata = [[(sum(left for left, _ in group), sum(right for _, right in group))
               for group in prompts.values()] for prompts in per_suite.values()]
    draws = []
    for _ in range(resamples):
        sampled = [pair for groups in strata for pair in rng.choices(groups, k=len(groups))]
        draws.append(sum(left for left, _ in sampled) / sum(right for _, right in sampled))

    def bytes_per_token(rows):
        return sum(row["disk_read_bytes"] for row in rows) / sum(row["tokens"] for row in rows)

    before, after = map(bytes_per_token, (baseline, candidate))
    lcb = sorted(draws)[int(resamples * 0.05)]
    speedup = sum(row["seconds"] for row in baseline) / sum(row["seconds"] for row in candidate)
    # At token parity, this paired time ratio equals the pooled tokens/sec ratio.
    return {
        "exact_greedy_q4_parity": equal,
        "paired_cells": len(oracle),
        "speedup": speedup if equal else None,
        "speedup_lcb95": lcb if equal else None,
        "baseline_disk_read_bytes_per_token": before,
        "candidate_disk_read_bytes_per_token": after,
        "passed": equal and lcb >= 3 and after < before,
    }


class FixtureTarget:
    def __init__(self, pager):
        self.pager = pager

    def reset(self):
        self.state = 0

    def prefetch(self, proposals):
        if self.pager.rung in (4, 6):
            self.pager.fetch([p % 12 for p in proposals], prefetch=True)

    def start(self, prompt):
        self.state = sum(prompt)
        return self.advance(prompt[-1])

    def advance(self, token):
        keys = [(token + 1) % 12, token % 12, (token + 2) % 12]
        if self.pager.rung == 5:
            keys = [key for key in keys if key != 11]
        blocks = self.pager.fetch(keys)
        self.state = (self.state + token + sum(blocks[key][0] for key in keys)) % 113
        return self.state


class FixtureDraft:
    def reset(self):
        self.round = 0

    def propose(self, prefix, width):
        # Its first round uses an independently specified correct fixture token.
        first = (sum(prefix) + prefix[-1] + sum((prefix[-1] + i) % 12 for i in range(3))) % 113
        self.round += 1
        return ([first] + [0] * (width - 1)) if self.round == 1 else [0] * width


def drycheck(directory):
    path = Path(directory) / "experts.bin"
    block_size = 4096
    path.write_bytes(b"".join(bytes([i]) * block_size for i in range(12)))
    expected_digest = file_digest(path)
    compressed_path = Path(directory) / "compressed-experts.bin"
    compressed_path.write_bytes(b"".join(bytes([i // 3 * 3]) * block_size for i in range(12)))
    extents = {i: Extent(i * block_size, block_size) for i in range(12)}
    prompts = ([4, 9], [31, 22, 1], [19, 1, 4], [3, 2])
    reports = {}
    reference_sequences = []
    for rung in range(1, 7):
        with ExpertPager(
            compressed_path if rung == 5 else path, extents, rung, cache_bytes=3 * block_size if rung > 1 else 0,
            pin_bytes=2 * block_size if rung == 6 else 0,
        ) as pager:
            sequences = []
            rows = []
            for prompt in prompts:
                pager.reset(cold=True)
                draft = FixtureDraft() if rung in (4, 6) else None
                start = time.perf_counter()
                tokens, proposed, accepted = greedy_decode(FixtureTarget(pager), prompt, max_tokens=18, draft=draft)
                rows.append({
                    "seconds": time.perf_counter() - start,
                    "read_calls": pager.read_calls, "requested_read_bytes": pager.bytes_read,
                    "proposed": proposed, "accepted": accepted,
                })
                sequences.append(tokens)
            arm = {"sequence_sha256": json_digest(sequences), "rows": rows}
            if rung == 1:
                reference_sequences = sequences
            if rung == 5:
                arm["separate_fixture_quality"] = [
                    reference_quality(candidate, reference) for candidate, reference in zip(sequences, reference_sequences)
                ]
            reports[RUNGS[rung - 1]] = arm
    parity = len({reports[RUNGS[rung - 1]]["sequence_sha256"] for rung in LOSSLESS_RUNGS}) == 1
    draft_rows = reports[RUNGS[3]]["rows"] + reports[RUNGS[5]]["rows"]
    verified = sum(row["accepted"] for row in draft_rows)
    rejected = sum(row["proposed"] - row["accepted"] for row in draft_rows)
    return {
        "mode": "local-drycheck", "fixture_sha256": expected_digest, "extent_count": len(extents),
        "compressed_fixture_sha256": file_digest(compressed_path),
        "rungs": reports, "exact_fixture_parity": parity, "accepted_draft_tokens": verified,
        "unaccepted_draft_tokens": rejected,
        "io_method": "pread" if hasattr(os, "pread") else "offset_readfile",
        "passed": bool(parity and verified and rejected and file_digest(path) == expected_digest),
    }


def reference_quality(tokens, reference):
    """Score reference continuation overlap separately from Q4 parity and throughput."""
    import difflib

    return {
        "reference_exact_match": tokens == reference,
        "reference_token_similarity": difflib.SequenceMatcher(None, reference, tokens, autojunk=False).ratio(),
    }


def summarize_quality(rows):
    return {
        suite: {
            "runs": len(cells),
            "reference_exact_match_rate": statistics.mean(c["quality"]["reference_exact_match"] for c in cells),
            "mean_reference_token_similarity": statistics.mean(c["quality"]["reference_token_similarity"] for c in cells),
        }
        for suite in sorted({row["suite"] for row in rows})
        if (cells := [row for row in rows if row["suite"] == suite])
    }
 

def tensor_storage_bytes(module):
    storages = {}
    for tensor in tuple(module.parameters()) + tuple(module.buffers()):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


class QwenDraft:
    """Incremental CPU Qwen3-0.6B draft with KV rollback after target rejection."""

    def __init__(self, path, tokenizer, max_context):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        draft_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if draft_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Target and draft token IDs must use the same vocabulary")
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map=None, use_safetensors=True,
        ).eval()
        config = self.model.config
        if (config.model_type, config.hidden_size, config.num_hidden_layers, config.num_attention_heads) != (
            "qwen3", 1024, 28, 16,
        ):
            raise ValueError("Expected the Qwen3-0.6B draft architecture")
        # Materialize file-backed safetensors into resident CPU storage before the first timed cell.
        with torch.no_grad():
            for tensor in tuple(self.model.parameters()) + tuple(self.model.buffers()):
                tensor.data = tensor.detach().clone()
        self.weight_bytes = tensor_storage_bytes(self.model)
        self.kv_max_bytes = 4 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * max_context
        self.torch = torch

    def reset(self):
        from transformers import DynamicCache

        self.prefix = []
        self.cache = DynamicCache()

    def propose(self, prefix, width):
        # Recompute the last token to recover its logits when all prior proposals were verified.
        common = 0
        for expected, cached in zip(prefix[:-1], self.prefix):
            if expected != cached:
                break
            common += 1
        self.cache.crop(common)
        tokens = list(prefix[common:])
        proposals = []
        with self.torch.inference_mode():
            for _ in range(width):
                result = self.model(input_ids=self.torch.tensor([tokens]), past_key_values=self.cache,
                                    use_cache=True, logits_to_keep=1)
                self.cache = result.past_key_values
                token = int(result.logits[0, -1].argmax())
                proposals.append(token)
                tokens = [token]
        self.prefix = list(prefix) + proposals[:-1]
        return proposals


@dataclass(frozen=True)
class ExpertTensor:
    qtype: object
    packed_row_bytes: int
    out_features: int


class QwenDiskTarget:
    """Qwen3-MoE GGUF forward pass with per-expert quantized disk reads and CPU KV."""

    def __init__(self, path, *, rung, cache_bytes, pin_bytes, context, reap=None, physical_offset=None):
        import numpy as np
        import torch
        from gguf import GGUFReader
        from gguf.quants import dequantize

        self.np = np
        self.torch = torch
        self.f = torch.nn.functional
        self.dequantize = dequantize
        self.context = context
        self.rung = rung
        self.reap = reap
        reader = GGUFReader(path)
        fields = reader.fields

        def scalar(name):
            field = fields[name]
            value = field.parts[field.data[0]]
            return bytes(value).decode() if field.types[-1].name == "STRING" else value.item()

        arch = scalar("general.architecture")
        if arch != "qwen3moe":
            raise ValueError("The target must be a Qwen3 MoE GGUF")
        self.layers = int(scalar(f"{arch}.block_count"))
        self.n_experts = int(scalar(f"{arch}.expert_count"))
        self.top_experts = int(scalar(f"{arch}.expert_used_count"))
        self.heads = int(scalar(f"{arch}.attention.head_count"))
        self.kv_heads = int(scalar(f"{arch}.attention.head_count_kv"))
        self.head_dim = int(scalar(f"{arch}.rope.dimension_count"))
        self.width = int(scalar(f"{arch}.embedding_length"))
        self.eps = float(scalar(f"{arch}.attention.layer_norm_rms_epsilon"))
        self.rope_base = float(scalar(f"{arch}.rope.freq_base"))
        self.scale = self.head_dim ** -0.5
        self.metadata = {
            "architecture": arch, "layers": self.layers, "experts": self.n_experts, "hidden_size": self.width,
            "top_experts": self.top_experts, "heads": self.heads, "kv_heads": self.kv_heads, "head_dim": self.head_dim,
            "rope_base": self.rope_base, "rms_eps": self.eps,
            "expert_weights_normalized": (
                bool(scalar(f"{arch}.expert_weights_norm")) if f"{arch}.expert_weights_norm" in fields else True
            ),
        }
        if (self.layers, self.n_experts, self.top_experts, self.width, self.heads, self.kv_heads, self.head_dim) != (
            48, 128, 8, 2048, 32, 4, 128,
        ):
            raise ValueError("The GGUF dimensions do not identify Qwen3-30B-A3B")
        if f"{arch}.rope.scaling.type" in fields and scalar(f"{arch}.rope.scaling.type") != "none":
            raise ValueError("Only the stock Qwen3 rope configuration is supported")
        if reap is not None and (
            set(reap) != set(range(self.layers))
            or any(len(ids) < self.top_experts or not ids <= set(range(self.n_experts)) for ids in reap.values())
            or all(len(ids) == self.n_experts for ids in reap.values())
        ):
            raise ValueError("REAP mask must retain at least top-k experts per layer and prune some experts")

        extents = {}
        self.specs = {}
        self.weights = {}
        expert_quantizations = set()
        with torch.inference_mode():
            for tensor in reader.tensors:
                size = int(tensor.n_bytes)
                name = tensor.name
                if name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")):
                    # The contiguous GGUF tensor order is [expert, output row, packed input row].
                    _, out_features, experts = map(int, tensor.shape)
                    if experts != self.n_experts or size % (experts * out_features):
                        raise ValueError(f"Unsupported packed expert tensor geometry: {name}")
                    per_expert = size // experts
                    expert_quantizations.add(tensor.tensor_type.name)
                    self.specs[name] = ExpertTensor(tensor.tensor_type, per_expert // out_features, out_features)
                    for expert in range(experts):
                        extents[(name, expert)] = Extent(int(tensor.data_offset) + expert * per_expert, per_expert)
                else:
                    # Decode one row block at a time so embedding/output conversion cannot spike the cage.
                    rows = tensor.data
                    output_shape = tuple(int(d) for d in reversed(tensor.shape))
                    weight = torch.empty(output_shape, dtype=torch.bfloat16)
                    if len(output_shape) == 1:
                        weight.copy_(torch.from_numpy(dequantize(rows, tensor.tensor_type).copy()))
                    else:
                        for start in range(0, len(rows), 256):
                            chunk = dequantize(rows[start:start + 256], tensor.tensor_type)
                            weight[start:start + len(chunk)].copy_(torch.from_numpy(np.array(chunk, copy=True)))
                    self.weights[name] = weight
        if len(self.specs) != 3 * self.layers or "token_embd.weight" not in self.weights:
            raise ValueError("Missing Qwen3 target weights")
        if rung == 5:
            if reap is None or not any(quant.startswith("IQ2_") for quant in expert_quantizations):
                raise ValueError("R5 requires IQ2 expert weights and an explicit REAP pruning mask")
        elif not expert_quantizations or not all(quant.startswith("Q4_") for quant in expert_quantizations):
            raise ValueError("Lossless rungs require the same Q4 GGUF expert quantization")
        self.metadata["expert_quantizations"] = sorted(expert_quantizations)
        self.weight_bytes = sum(weight.numel() * weight.element_size() for weight in self.weights.values())
        self.kv_bytes = 4 * self.layers * self.kv_heads * self.head_dim * context
        self.read_working_bytes = 16 * MIB + sum(sorted((e.size for e in extents.values()), reverse=True)[:3 * self.top_experts])
        del reader
        self.pager = ExpertPager(
            path, extents, rung, cache_bytes=cache_bytes, pin_bytes=pin_bytes, direct=rung != 1,
            physical_offset=physical_offset,
        )
        self.vocab = self.weights["token_embd.weight"].shape[0]
        self.routes = OrderedDict()
        self.max_route_tokens = 64
        self.keys = torch.empty((self.layers, context, self.kv_heads, self.head_dim), dtype=torch.bfloat16)
        self.values = torch.empty_like(self.keys)
        self.rotary_inverse = self.rope_base ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)

    def reset(self):
        self.position = 0
        self.routes.clear()

    def prefetch(self, proposals):
        keys = []
        for token in proposals:
            for key in self.routes.get(token, ()):
                if key not in keys:
                    keys.append(key)
        # A speculative miss cannot evict an entire demand cache.
        max_bytes = self.pager.cache_bytes // 4
        chosen = []
        for key in keys:
            size = self.pager.extents[key].size
            if size > max_bytes:
                break
            max_bytes -= size
            chosen.append(key)
        self.pager.fetch(chosen, prefetch=True)

    def _linear(self, value, name):
        return self.f.linear(value, self.weights[name])

    def _norm(self, value, name):
        fp = value.float()
        return (fp * (fp.square().mean(dim=-1, keepdim=True) + self.eps).rsqrt()).to(value.dtype) * self.weights[name]

    def _rotary(self, value, cos, sin):
        left, right = value.float().chunk(2, dim=-1)
        return self.torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1).to(value.dtype)

    def _attention(self, value, layer, name, cos, sin):
        q = self._linear(value, name + "attn_q.weight").view(self.heads, self.head_dim)
        k = self._linear(value, name + "attn_k.weight").view(self.kv_heads, self.head_dim)
        v = self._linear(value, name + "attn_v.weight").view(self.kv_heads, self.head_dim)
        q = self._rotary(self._norm(q, name + "attn_q_norm.weight"), cos, sin)
        k = self._rotary(self._norm(k, name + "attn_k_norm.weight"), cos, sin)
        self.keys[layer, self.position].copy_(k)
        self.values[layer, self.position].copy_(v)
        upper = self.position + 1
        groups = self.heads // self.kv_heads
        q = q.view(self.kv_heads, groups, self.head_dim).float()
        keys = self.keys[layer, :upper].transpose(0, 1).float()
        values = self.values[layer, :upper].transpose(0, 1).float()
        probs = (q @ keys.transpose(1, 2) * self.scale).softmax(dim=-1)
        result = (probs @ values).flatten().to(v.dtype)
        return self._linear(result, name + "attn_output.weight")

    def _expert_linear(self, vector, key, data):
        spec = self.specs[key[0]]
        raw = self.np.frombuffer(data, dtype=self.np.uint8).reshape(spec.out_features, spec.packed_row_bytes)
        decoded = self.dequantize(raw, spec.qtype)
        weight = self.torch.from_numpy(decoded).to(dtype=vector.dtype)
        return self.f.linear(vector, weight)

    def _moe(self, vector, layer, name):
        router_logits = self._linear(vector, name + "ffn_gate_inp.weight").float()
        if self.reap is not None:
            removed = sorted(set(range(self.n_experts)) - self.reap[layer])
            router_logits[removed] = float("-inf")
        probs, experts = router_logits.softmax(dim=-1).topk(self.top_experts)
        if self.metadata["expert_weights_normalized"]:
            probs = probs / probs.sum()
        projections = [name + f"ffn_{proj}_exps.weight" for proj in ("gate", "up", "down")]
        keys = [tuple((projection, expert) for projection in projections) for expert in experts.tolist()]
        load_order = [key for group in keys for key in group]
        blocks = self.pager.fetch(load_order)
        result = self.torch.zeros_like(vector, dtype=self.torch.float32)
        for prob, (gate, up, down) in zip(probs, keys):
            hidden = self.f.silu(self._expert_linear(vector, gate, blocks[gate]))
            hidden = hidden * self._expert_linear(vector, up, blocks[up])
            result.add_(self._expert_linear(hidden, down, blocks[down]).float() * prob)
        return result.to(vector.dtype), load_order

    def _advance(self, token, *, logits):
        if not 0 <= token < self.vocab or self.position == self.context:
            raise ValueError("Token or position exceeds the model context")
        value = self.weights["token_embd.weight"][token]
        theta = self.position * self.rotary_inverse
        cos, sin = theta.cos(), theta.sin()
        routed = []
        for layer in range(self.layers):
            name = f"blk.{layer}."
            attn = self._norm(value, name + "attn_norm.weight")
            value = value + self._attention(attn, layer, name, cos, sin)
            mlp = self._norm(value, name + "ffn_norm.weight")
            output, routes = self._moe(mlp, layer, name)
            value = value + output
            routed.extend(routes)
        self.position += 1
        if self.rung in (4, 6):
            self.routes[token] = routed
            self.routes.move_to_end(token)
            if len(self.routes) > self.max_route_tokens:
                self.routes.popitem(last=False)
        if logits:
            value = self._norm(value, "output_norm.weight")
            scores = self.f.linear(value, self.weights.get("output.weight", self.weights["token_embd.weight"]))
            return int(scores.argmax())

    def start(self, prompt):
        with self.torch.inference_mode():
            for token in prompt[:-1]:
                self._advance(token, logits=False)
            return self._advance(prompt[-1], logits=True)

    def advance(self, token):
        with self.torch.inference_mode():
            return self._advance(token, logits=True)
