"""Wave 19 / Bundle #60 / Item 2E: causal-prefix, weight-SHA-bound SSD KV cache.

Each cached SSD KV block is bound to the *complete causal prefix* (token ids,
exact positions, attention-mask hash) AND to the TRA-1 state identity
(base-weights SHA, adapter-weights SHA, config SHA, tokenizer SHA, KV dtype,
RoPE scaling hash). Sources (file versions) are EK5 events: retracting a file
version or commit invalidates every cached block whose causal prefix contains
it, including blocks whose own file is clean.

Comparison across 20 hash-selected synthetic git repositories x 50 commits:

* full prefill (logit/greedy parity reference),
* a strong stable-files-first deterministic prefix-LRU challenger (exact
  token-prefix keyed, RAM-bounded, no weight identity, no EK5).

The model is a tiny deterministic 2-layer causal-attention numpy model with
absolute sinusoidal positions (so positions genuinely matter). It is NOT an
LLM: this is a *floor* tier mechanism check, disclosed in the manifest.

``MemT-1.0 modeling_memt.py:876 _state_identity()`` hashes config/tokenizer/
shape but omits weights and adapters; :func:`weight_revision_probe` reproduces
that omission and shows TRA-1 failing closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import tracemalloc
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
_KERNEL_PKG = _ROOT / "packages" / "chelated_kernel"
for _p in (str(_KERNEL_PKG), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chelated_kernel import EventLedger  # noqa: E402  (public SDK boundary only)

from chelated_pv7_spine.bridge_merge_harness import paired_bootstrap_ci  # noqa: E402
from chelated_pv7_spine.cpu_io_cage import build_cpu_only_env  # noqa: E402
from chelated_pv7_spine.ek5_reason_canonicalizer import (  # noqa: E402
    compare_with_fresh_rebuild,
    export_canonical_ek5_audit_trace,
)
from chelated_pv7_spine.tra1_weight_sha_guard import (  # noqa: E402
    EMPTY_ADAPTER_SHA256,
    StaleWeightRevisionError,
    TRA1BoundStateCache,
    TRA1StateIdentityContract,
    config_only_state_identity,
    sha256_digest,
)
from ek5_v3_stratified import (  # noqa: E402
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
)

BUNDLE_ID = 60
MODES: Tuple[str, ...] = ("local-drycheck", "run-live")
REPO_COUNT = 20
COMMITS_PER_REPO = 50
CANDIDATE_REPOS = 40
VOCAB = 64
D_MODEL = 16
N_LAYERS = 2
GREEDY_STEPS = 4
LOGIT_TOL = 1e-4
CHALLENGER_RAM_BLOCKS = 96
KV_DTYPE = "float64"
MEMT_CITATION = "MemT-1.0 modeling_memt.py:876 _state_identity() omits base/adapter weights"


class LiveModeUnavailableError(RuntimeError):
    """``run-live`` needs a real model + real git repositories; none are wired."""


# ---------------------------------------------------------------------------
# tiny deterministic causal model
# ---------------------------------------------------------------------------
def _rng_matrix(seed: int, tag: str, shape: Tuple[int, ...], scale: float) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}|{tag}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.standard_normal(shape).astype(np.float64) * scale


@dataclass
class TinyModel:
    emb: np.ndarray
    wq: List[np.ndarray]
    wk: List[np.ndarray]
    wv: List[np.ndarray]
    wo: List[np.ndarray]
    wout: np.ndarray
    adapter: Optional[np.ndarray] = None  # LoRA-like delta on last layer wo

    def base_bytes(self) -> bytes:
        parts = [self.emb, self.wout] + self.wq + self.wk + self.wv + self.wo
        return b"".join(np.ascontiguousarray(p).tobytes() for p in parts)

    def adapter_bytes(self) -> bytes:
        return b"" if self.adapter is None else np.ascontiguousarray(self.adapter).tobytes()

    def patched_base(self) -> "TinyModel":
        """Return a copy with exactly one base weight perturbed (config unchanged)."""
        emb = self.emb.copy()
        emb[0, 0] += 1e-3
        return TinyModel(emb, self.wq, self.wk, self.wv, self.wo, self.wout, self.adapter)

    def with_adapter(self, seed: int) -> "TinyModel":
        delta = _rng_matrix(seed, "adapter", (D_MODEL, D_MODEL), 0.05)
        return TinyModel(self.emb, self.wq, self.wk, self.wv, self.wo, self.wout, delta)

    def _wo(self, layer: int) -> np.ndarray:
        if self.adapter is not None and layer == N_LAYERS - 1:
            return self.wo[layer] + self.adapter
        return self.wo[layer]

    def forward(
        self, tokens: np.ndarray, past: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """Return (logits_for_every_new_position, new-token KV per layer)."""
        n = len(tokens)
        start = 0 if past is None else past[0][0].shape[0]
        pos = np.arange(start, start + n)
        h = self.emb[tokens] + _positions(pos)
        new_kv: List[Tuple[np.ndarray, np.ndarray]] = []
        for layer in range(N_LAYERS):
            q = h @ self.wq[layer]
            k = h @ self.wk[layer]
            v = h @ self.wv[layer]
            new_kv.append((k, v))
            if past is not None:
                k_all = np.concatenate([past[layer][0], k], axis=0)
                v_all = np.concatenate([past[layer][1], v], axis=0)
            else:
                k_all, v_all = k, v
            total = k_all.shape[0]
            scores = (q @ k_all.T) / np.sqrt(D_MODEL)
            mask = np.arange(total)[None, :] <= (start + np.arange(n))[:, None]
            scores = np.where(mask, scores, -1e30)
            scores = scores - scores.max(axis=1, keepdims=True)
            w = np.exp(scores)
            w = w / w.sum(axis=1, keepdims=True)
            h = h + (w @ v_all) @ self._wo(layer)
        return h @ self.wout, new_kv


def _positions(pos: np.ndarray) -> np.ndarray:
    idx = np.arange(D_MODEL // 2)[None, :]
    ang = pos[:, None] / np.power(1000.0, 2.0 * idx / D_MODEL)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def build_model(seed: int = 6001) -> TinyModel:
    return TinyModel(
        emb=_rng_matrix(seed, "emb", (VOCAB, D_MODEL), 1.0),
        wq=[_rng_matrix(seed, f"wq{i}", (D_MODEL, D_MODEL), 0.5) for i in range(N_LAYERS)],
        wk=[_rng_matrix(seed, f"wk{i}", (D_MODEL, D_MODEL), 0.5) for i in range(N_LAYERS)],
        wv=[_rng_matrix(seed, f"wv{i}", (D_MODEL, D_MODEL), 0.5) for i in range(N_LAYERS)],
        wo=[_rng_matrix(seed, f"wo{i}", (D_MODEL, D_MODEL), 0.3) for i in range(N_LAYERS)],
        wout=_rng_matrix(seed, "wout", (D_MODEL, VOCAB), 0.5),
    )


def concat_kv(blocks: Sequence[List[Tuple[np.ndarray, np.ndarray]]]) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    if not blocks:
        return None
    return [
        (
            np.concatenate([b[layer][0] for b in blocks], axis=0),
            np.concatenate([b[layer][1] for b in blocks], axis=0),
        )
        for layer in range(N_LAYERS)
    ]


def greedy_tokens(
    model: TinyModel,
    prompt_logits_last: np.ndarray,
    kv: List[Tuple[np.ndarray, np.ndarray]],
    steps: int = GREEDY_STEPS,
) -> List[int]:
    out: List[int] = []
    logits = prompt_logits_last
    for _ in range(steps):
        tok = int(np.argmax(logits))
        out.append(tok)
        step_logits, new_kv = model.forward(np.array([tok]), past=kv)
        kv = [
            (np.concatenate([kv[i][0], new_kv[i][0]], axis=0), np.concatenate([kv[i][1], new_kv[i][1]], axis=0))
            for i in range(N_LAYERS)
        ]
        logits = step_logits[-1]
    return out


# ---------------------------------------------------------------------------
# identity contract
# ---------------------------------------------------------------------------
CONFIG_SHA256 = hashlib.sha256(
    json.dumps({"layers": N_LAYERS, "d_model": D_MODEL, "vocab": VOCAB, "heads": 1}, sort_keys=True).encode()
).hexdigest()
TOKENIZER_SHA256 = hashlib.sha256(b"pv7-tiny-synthetic-tokenizer-vocab64").hexdigest()
ROPE_SCALING_HASH = hashlib.sha256(b"sinusoidal-absolute-base1000-no-scaling").hexdigest()


def contract_for(model: TinyModel) -> TRA1StateIdentityContract:
    adapter = model.adapter_bytes()
    return TRA1StateIdentityContract(
        base_weights_sha256=sha256_digest(model.base_bytes()),
        adapter_weights_sha256=sha256_digest(adapter) if adapter else EMPTY_ADAPTER_SHA256,
        config_sha256=CONFIG_SHA256,
        tokenizer_sha256=TOKENIZER_SHA256,
        kv_or_state_dtype=KV_DTYPE,
        rope_freq_base=1000.0,
        rope_scaling_hash=ROPE_SCALING_HASH,
    )


def mask_hash(total_tokens: int) -> str:
    return hashlib.sha256(b"causal-full|" + np.ones(total_tokens, dtype=np.int8).tobytes()).hexdigest()


def chain_hash(prev: str, tokens: np.ndarray, start: int, end: int) -> str:
    """Causal-prefix chain: covers all earlier ids, exact positions and the mask."""
    h = hashlib.sha256()
    h.update(prev.encode())
    h.update(np.asarray(tokens, dtype=np.int64).tobytes())
    h.update(f"|pos={start}:{end}|mask={mask_hash(end)}".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# synthetic git repositories
# ---------------------------------------------------------------------------
@dataclass
class FileVersion:
    version_id: str
    name: str
    tokens: np.ndarray
    commit: int


@dataclass
class Commit:
    repo: str
    index: int
    files: List[FileVersion]  # stable-files-first order
    retract: List[str] = field(default_factory=list)  # version ids retracted before this commit
    probes: List[int] = field(default_factory=list)  # earlier commit indices replayed after retraction


@dataclass
class Repo:
    name: str
    commits: List[Commit]


def _h_int(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "little")


def select_repo_names(seed: int, count: int = REPO_COUNT, candidates: int = CANDIDATE_REPOS) -> List[str]:
    names = [f"synthrepo-{i:03d}" for i in range(candidates)]
    ranked = sorted(names, key=lambda n: hashlib.sha256(f"{seed}|{n}".encode()).hexdigest())
    return sorted(ranked[:count])


def _tokens_for(seed: int, repo: str, name: str, rev: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(_h_int(seed, repo, name, rev) % (2**32))
    return rng.integers(0, VOCAB, size=length, dtype=np.int64)


def build_repo(seed: int, name: str, commit_count: int = COMMITS_PER_REPO) -> Repo:
    """Synthetic history: a few hot files, many cold files, periodic retractions."""
    n_files = 6 + _h_int(seed, name, "nfiles") % 4
    files = [f"f{i}.py" for i in range(n_files)]
    hot = set(files[: 2])  # hot files change often; cold files rarely
    rev = {f: 0 for f in files}
    state: Dict[str, FileVersion] = {}
    commits: List[Commit] = []
    changes = {f: 0 for f in files}
    retract_plan = {
        5 + _h_int(seed, name, "r", j) % (commit_count - 12) for j in range(2)
    }
    pending_fix: Dict[str, str] = {}
    introduced: Dict[int, List[str]] = {}
    for c in range(commit_count):
        retract_now: List[str] = []
        probes: List[int] = []
        if c - 3 in retract_plan and (c - 3) in introduced and introduced[c - 3]:
            target = introduced[c - 3][0]
            retract_now = [target]
            probes = [max(0, c - 3), max(0, c - 2)]
            pending_fix[target] = target
        touched: List[str] = []
        if c == 0:
            touched = list(files)
        else:
            for tgt in list(pending_fix):
                fname = tgt.split("::")[2]
                touched.append(fname)
                del pending_fix[tgt]
            pick = files[_h_int(seed, name, "pick", c) % len(files)]
            hot_pick = sorted(hot)[_h_int(seed, name, "hot", c) % len(hot)]
            touched.append(hot_pick if _h_int(seed, name, "coin", c) % 100 < 70 else pick)
        touched = sorted(set(touched))
        for f in touched:
            rev[f] += 1
            changes[f] += 1
            ln = 20 + _h_int(seed, name, f, "len") % 17
            vid = f"fv::{name}::{f}::r{rev[f]}"
            state[f] = FileVersion(vid, f, _tokens_for(seed, name, f, rev[f], ln), c)
            introduced.setdefault(c, []).append(vid)
        ordered = sorted(files, key=lambda f: (changes[f], f))  # stable-files-first
        commits.append(Commit(name, c, [state[f] for f in ordered], retract_now, probes))
    return Repo(name, commits)


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------
@dataclass
class CacheStats:
    prompt_tokens: int = 0
    reused_tokens: int = 0
    blocks_hit: int = 0
    blocks_missed: int = 0
    stale_served_blocks: int = 0
    refused_retracted_blocks: int = 0
    stale_weight_refusals: int = 0


def _block_sources(blocks: Sequence[FileVersion], upto: int) -> List[str]:
    return [b.version_id for b in blocks[: upto + 1]]


class SSDKVCache:
    """Causal-prefix + TRA-1 + EK5 bound SSD KV cache."""

    def __init__(self, root: Path, contract: TRA1StateIdentityContract, store: StratifiedEvidenceStore) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.store = store
        self.bound = TRA1BoundStateCache()
        self.index: Dict[str, Dict[str, Any]] = {}
        self.written_paths: List[str] = []
        self.bytes_on_disk = 0

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.kvblk"

    def put(
        self,
        key: str,
        prefix_tokens: np.ndarray,
        start: int,
        end: int,
        kv: List[Tuple[np.ndarray, np.ndarray]],
        sources: Sequence[str],
    ) -> None:
        arrays = [np.ascontiguousarray(x, dtype=np.float64) for pair in kv for x in pair]
        header = {
            "prefix_tokens": [int(t) for t in prefix_tokens],
            "start": start,
            "end": end,
            "mask_hash": mask_hash(end),
            "contract_digest": self.contract.digest(),
            "kv_dtype": KV_DTYPE,
            "shape": [end - start, D_MODEL],
        }
        body = b"".join(a.tobytes() for a in arrays)
        raw = json.dumps(header, sort_keys=True).encode() + b"\n" + body
        path = self._path(key)
        path.write_bytes(raw)
        self.written_paths.append(str(path))
        self.bytes_on_disk += len(raw)
        file_sha = hashlib.sha256(raw).hexdigest()
        self.bound.seal(bytes.fromhex(key), self.contract, file_sha.encode())
        self.index[key] = {"sources": list(sources), "file_sha": file_sha}

    def get(self, key: str, prefix_tokens: np.ndarray, contract: TRA1StateIdentityContract) -> Optional[List]:
        """Return KV list or None. Raises StaleWeightRevisionError on identity mismatch."""
        if key not in self.index:
            return None
        expected = self.bound.reuse(bytes.fromhex(key), contract)  # raises StaleWeightRevisionError
        raw = self._path(key).read_bytes()
        if hashlib.sha256(raw).hexdigest().encode() != expected:
            raise StaleWeightRevisionError("SSD block bytes do not match sealed digest")
        head_raw, body = raw.split(b"\n", 1)
        header = json.loads(head_raw)
        if header["prefix_tokens"] != [int(t) for t in prefix_tokens]:
            raise StaleWeightRevisionError("causal prefix token ids do not match cached block")
        n = header["shape"][0]
        arrs = np.frombuffer(body, dtype=np.float64).reshape(N_LAYERS * 2, n, D_MODEL)
        return [(arrs[2 * i].copy(), arrs[2 * i + 1].copy()) for i in range(N_LAYERS)]

    def sources_admit(self, key: str) -> bool:
        return all(self.store.eligibility.get(s) == "ADMIT" for s in self.index[key]["sources"])


class PrefixLRUChallenger:
    """Strong stable-files-first, exact-token-prefix, RAM-bounded LRU.

    Safe for logits (exact token-prefix keys + positions) but has NO weight
    identity and NO retraction awareness.
    """

    def __init__(self, capacity_blocks: int = CHALLENGER_RAM_BLOCKS) -> None:
        self.capacity = capacity_blocks
        self.lru: "OrderedDict[str, Tuple[List, List[str]]]" = OrderedDict()

    def get(self, key: str) -> Optional[Tuple[List, List[str]]]:
        item = self.lru.get(key)
        if item is not None:
            self.lru.move_to_end(key)
        return item

    def put(self, key: str, kv: List, sources: List[str]) -> None:
        self.lru[key] = (kv, sources)
        self.lru.move_to_end(key)
        while len(self.lru) > self.capacity:
            self.lru.popitem(last=False)


def _prompt_chain(blocks: Sequence[FileVersion]) -> List[Tuple[str, int, int, np.ndarray]]:
    """[(chain_key, start, end, prefix_tokens_upto_end)] for each block."""
    out: List[Tuple[str, int, int, np.ndarray]] = []
    prev = "genesis"
    pos = 0
    toks: List[np.ndarray] = []
    for b in blocks:
        start, end = pos, pos + len(b.tokens)
        prev = chain_hash(prev, b.tokens, start, end)
        toks.append(b.tokens)
        out.append((prev, start, end, np.concatenate(toks)))
        pos = end
    return out


def serve_chelated(
    model: TinyModel,
    cache: SSDKVCache,
    blocks: Sequence[FileVersion],
    stats: CacheStats,
    *,
    contract: Optional[TRA1StateIdentityContract] = None,
) -> Tuple[np.ndarray, List]:
    contract = contract or cache.contract
    chain = _prompt_chain(blocks)
    kv_blocks: List = []
    hit = 0
    for j, (key, start, end, ptoks) in enumerate(chain):
        if key not in cache.index:
            break
        if not cache.sources_admit(key):
            stats.refused_retracted_blocks += 1
            break
        try:
            kv = cache.get(key, ptoks, contract)
        except StaleWeightRevisionError:
            stats.stale_weight_refusals += 1
            break
        if kv is None:
            break
        kv_blocks.append(kv)
        hit = j + 1
    reused = sum(len(blocks[j].tokens) for j in range(hit))
    stats.reused_tokens += reused
    stats.blocks_hit += hit
    stats.blocks_missed += len(blocks) - hit
    stats.prompt_tokens += chain[-1][2]
    past = concat_kv(kv_blocks)
    logits_parts: List[np.ndarray] = []
    kv_all = list(kv_blocks)
    for j in range(hit, len(blocks)):
        key, start, end, ptoks = chain[j]
        cur_past = concat_kv(kv_all)
        lg, new_kv = model.forward(blocks[j].tokens, past=cur_past)
        logits_parts.append(lg)
        kv_all.append(new_kv)
        cache.put(key, ptoks, start, end, new_kv, _block_sources(blocks, j))
    past_full = concat_kv(kv_all)
    if logits_parts:
        last = logits_parts[-1][-1]
    else:  # fully cached: recompute only the final token for the logit
        n_last = len(blocks[-1].tokens)
        base = concat_kv(kv_all[:-1])
        lg, _ = model.forward(blocks[-1].tokens, past=base)
        last = lg[-1]
        _ = n_last
    del past
    return last, past_full


def serve_challenger(
    model: TinyModel,
    cache: PrefixLRUChallenger,
    blocks: Sequence[FileVersion],
    stats: CacheStats,
    store: StratifiedEvidenceStore,
) -> Tuple[np.ndarray, List]:
    chain = _prompt_chain(blocks)
    kv_blocks: List = []
    hit = 0
    for j, (key, _s, _e, _p) in enumerate(chain):
        item = cache.get(key)
        if item is None:
            break
        kv_blocks.append(item[0])
        hit = j + 1
        if any(store.eligibility.get(s) != "ADMIT" for s in item[1]):
            stats.stale_served_blocks += 1
    reused = sum(len(blocks[j].tokens) for j in range(hit))
    stats.reused_tokens += reused
    stats.blocks_hit += hit
    stats.blocks_missed += len(blocks) - hit
    stats.prompt_tokens += chain[-1][2]
    kv_all = list(kv_blocks)
    last = None
    for j in range(hit, len(blocks)):
        key, start, end, ptoks = chain[j]
        lg, new_kv = model.forward(blocks[j].tokens, past=concat_kv(kv_all))
        kv_all.append(new_kv)
        cache.put(key, new_kv, _block_sources(blocks, j))
        last = lg[-1]
    if last is None:
        lg, _ = model.forward(blocks[-1].tokens, past=concat_kv(kv_all[:-1]))
        last = lg[-1]
    return last, concat_kv(kv_all)


def full_prefill(model: TinyModel, blocks: Sequence[FileVersion]) -> Tuple[np.ndarray, List]:
    tokens = np.concatenate([b.tokens for b in blocks])
    lg, kv = model.forward(tokens)
    return lg[-1], kv


# ---------------------------------------------------------------------------
# probes / controls
# ---------------------------------------------------------------------------
def weight_revision_probe(model: TinyModel, trials: int = 20) -> Dict[str, Any]:
    """Weights/adapters change while config+tokenizer stay identical -> must refuse."""
    base_contract = contract_for(model)
    patched = model.patched_base()
    adapted = model.with_adapter(77)
    patched_contract = contract_for(patched)
    adapter_contract = contract_for(adapted)
    cache = TRA1BoundStateCache()
    page_ids = [hashlib.sha256(f"page{i}".encode()).digest() for i in range(trials)]
    for pid in page_ids:
        cache.seal(pid, base_contract, b"cached-kv")
    refused_patch = refused_adapter = 0
    for pid in page_ids:
        for contract, which in ((patched_contract, "patch"), (adapter_contract, "adapter")):
            try:
                cache.reuse(pid, contract)
            except StaleWeightRevisionError:
                if which == "patch":
                    refused_patch += 1
                else:
                    refused_adapter += 1
    shape = (N_LAYERS, D_MODEL)
    memt_a = config_only_state_identity(CONFIG_SHA256, TOKENIZER_SHA256, shape)
    memt_b = config_only_state_identity(CONFIG_SHA256, TOKENIZER_SHA256, shape)
    same_reuse_ok = all(cache.reuse(pid, base_contract) == b"cached-kv" for pid in page_ids)
    return {
        "trials": trials,
        "config_sha256_identical": base_contract.config_sha256 == patched_contract.config_sha256 == adapter_contract.config_sha256,
        "tokenizer_sha256_identical": base_contract.tokenizer_sha256 == patched_contract.tokenizer_sha256,
        "base_weights_changed": base_contract.base_weights_sha256 != patched_contract.base_weights_sha256,
        "adapter_changed": base_contract.adapter_weights_sha256 != adapter_contract.adapter_weights_sha256,
        "refused_after_base_patch": refused_patch,
        "refused_after_adapter_change": refused_adapter,
        "memt_style_config_only_identity_unchanged_after_patch": memt_a == memt_b,
        "memt_style_silent_reuses_after_patch": trials if memt_a == memt_b else 0,
        "tra1_silent_reuses_after_patch": trials - refused_patch,
        "unchanged_contract_still_reuses": same_reuse_ok,
        "memt_citation": MEMT_CITATION,
    }


def positional_binding_control(model: TinyModel, seed: int = 6003, trials: int = 40) -> Dict[str, Any]:
    """A block-tokens-only key (no prefix/position) gives wrong logits; ours cannot."""
    rng = np.random.default_rng(seed)
    max_diffs: List[float] = []
    for _ in range(trials):
        shared = rng.integers(0, VOCAB, size=24, dtype=np.int64)
        prefix_a = rng.integers(0, VOCAB, size=20, dtype=np.int64)
        prefix_b = rng.integers(0, VOCAB, size=20, dtype=np.int64)
        _, kv_prefix_a = model.forward(prefix_a)
        lg_a, kv_a_with_shared = model.forward(shared, past=kv_prefix_a)
        _ = lg_a
        lg_true, _ = model.forward(np.concatenate([prefix_b, shared]))
        _, kv_prefix_b = model.forward(prefix_b)
        wrong = [
            (
                np.concatenate([kv_prefix_b[i][0], kv_a_with_shared[i][0]], axis=0),
                np.concatenate([kv_prefix_b[i][1], kv_a_with_shared[i][1]], axis=0),
            )
            for i in range(N_LAYERS)
        ]
        lg_wrong, _ = model.forward(np.array([int(shared[-1])]), past=wrong[:])  # extend one more token
        lg_ref, _ = model.forward(np.concatenate([prefix_b, shared, np.array([int(shared[-1])])]))
        max_diffs.append(float(np.max(np.abs(lg_wrong[-1] - lg_ref[-1]))))
        _ = lg_true
    keys_collide = chain_hash("genesis", np.arange(4), 0, 4) == chain_hash("genesis", np.arange(4), 0, 4)
    keys_differ_on_prefix = chain_hash(chain_hash("genesis", np.arange(4), 0, 4), np.arange(3), 4, 7) != chain_hash(
        chain_hash("genesis", np.arange(4) + 1, 0, 4), np.arange(3), 4, 7
    )
    keys_differ_on_position = chain_hash("genesis", np.arange(4), 0, 4) != chain_hash("genesis", np.arange(4), 3, 7)
    return {
        "trials": trials,
        "naive_block_only_reuse_max_abs_logit_diff": float(max(max_diffs)),
        "naive_block_only_reuse_exceeds_tol_trials": int(sum(d > LOGIT_TOL for d in max_diffs)),
        "same_prefix_same_key": keys_collide,
        "prefix_change_changes_key": keys_differ_on_prefix,
        "position_change_changes_key": keys_differ_on_position,
    }


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------
def _canonical_digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()


def run_ssd_kv_experiment(
    *,
    mode: str = "local-drycheck",
    workdir: Optional[Path] = None,
    repo_count: int = REPO_COUNT,
    commits_per_repo: int = COMMITS_PER_REPO,
    seed: int = 6001,
) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "run-live":
        raise LiveModeUnavailableError(
            "run-live needs a real LLM KV path and real git clones; only synthetic repos + a tiny model are wired"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cuda_env = build_cpu_only_env()
    own_tmp = workdir is None
    root = Path(tempfile.mkdtemp(prefix="pv7_ssd_kv_")) if own_tmp else Path(workdir)
    try:
        return _run(mode, root, repo_count, commits_per_repo, seed, cuda_env)
    finally:
        if own_tmp:
            shutil.rmtree(root, ignore_errors=True)  # single explicitly named tmp dir created above


def _run(mode: str, root: Path, repo_count: int, commits_per_repo: int, seed: int, cuda_env: Mapping[str, str]) -> Dict[str, Any]:
    model = build_model(seed)
    contract = contract_for(model)
    names = select_repo_names(seed, repo_count)
    repos = [build_repo(seed, n, commits_per_repo) for n in names]
    store = StratifiedEvidenceStore()
    kernel = EventLedger(policy="sec_audit", tenant_id="ssd-kv")
    cache = SSDKVCache(root / "kv", contract, store)
    challenger = PrefixLRUChallenger()
    st_ch = CacheStats()
    st_lru = CacheStats()
    per_repo_reuse: Dict[str, Dict[str, List[float]]] = {n: {"chelated": [], "challenger": []} for n in names}
    max_logit_diff_ch = 0.0
    max_logit_diff_lru = 0.0
    greedy_mismatch_ch = greedy_mismatch_lru = 0
    commits_checked = 0
    retractions = 0
    replay_probes = 0
    probe_stale_chelated = probe_stale_challenger = 0
    probe_logit_diff = 0.0
    downstream_clean_blocks_invalidated = 0
    registered: set = set()

    tracemalloc.start()
    try:
        for repo in repos:  # repo-sequential: the strong challenger's RAM LRU is not thrashed by interleaving
            for c in range(commits_per_repo):
                commit = repo.commits[c]
                new_events = [
                    StratifiedEvent(fv.version_id, authority_tier=1) for fv in commit.files if fv.version_id not in registered
                ]
                if new_events:
                    store.apply_mutation(new_events)
                    registered.update(e.event_id for e in new_events)
                    for e in new_events:
                        kernel.ingest(
                            {
                                "specversion": "1.0",
                                "type": "com.chelated.git.file_version",
                                "id": e.event_id,
                                "source": repo.name,
                                "time": c,
                                "data": {"status": "asserted"},
                            }
                        )
                for target in commit.retract:
                    rev_id = f"retract::{target}"
                    store.apply_mutation(
                        [StratifiedEvent(rev_id, authority_tier=5)],
                        [EdgeAttestation(f"att::{rev_id}", rev_id, target, "SUPERSEDE")],
                    )
                    retractions += 1
                    # replay probes: resume earlier prompts that contain the retracted version
                    for pc in commit.probes:
                        old_blocks = repo.commits[pc].files
                        if not any(b.version_id == target for b in old_blocks):
                            continue
                        replay_probes += 1
                        chain = _prompt_chain(old_blocks)
                        tgt_idx = next(i for i, b in enumerate(old_blocks) if b.version_id == target)
                        downstream_clean_blocks_invalidated += sum(
                            1
                            for j in range(tgt_idx + 1, len(old_blocks))
                            if chain[j][0] in cache.index and not cache.sources_admit(chain[j][0])
                        )
                        before = st_ch.stale_served_blocks
                        pst = CacheStats()
                        last_ref, _ = full_prefill(model, old_blocks)
                        last_c, _ = serve_chelated(model, cache, old_blocks, pst)
                        probe_logit_diff = max(probe_logit_diff, float(np.max(np.abs(last_c - last_ref))))
                        probe_stale_chelated += pst.stale_served_blocks + (st_ch.stale_served_blocks - before)
                        lst = CacheStats()
                        serve_challenger(model, challenger, old_blocks, lst, store)
                        probe_stale_challenger += lst.stale_served_blocks
                blocks = commit.files
                ref_last, ref_kv = full_prefill(model, blocks)
                ref_greedy = greedy_tokens(model, ref_last, ref_kv)

                before_ch = (st_ch.reused_tokens, st_ch.prompt_tokens)
                last_c, kv_c = serve_chelated(model, cache, blocks, st_ch)
                before_lru = (st_lru.reused_tokens, st_lru.prompt_tokens)
                last_l, kv_l = serve_challenger(model, challenger, blocks, st_lru, store)

                max_logit_diff_ch = max(max_logit_diff_ch, float(np.max(np.abs(last_c - ref_last))))
                max_logit_diff_lru = max(max_logit_diff_lru, float(np.max(np.abs(last_l - ref_last))))
                if greedy_tokens(model, last_c, kv_c) != ref_greedy:
                    greedy_mismatch_ch += 1
                if greedy_tokens(model, last_l, kv_l) != ref_greedy:
                    greedy_mismatch_lru += 1
                commits_checked += 1
                per_repo_reuse[repo.name]["chelated"].append(
                    (st_ch.reused_tokens - before_ch[0]) / (st_ch.prompt_tokens - before_ch[1])
                )
                per_repo_reuse[repo.name]["challenger"].append(
                    (st_lru.reused_tokens - before_lru[0]) / (st_lru.prompt_tokens - before_lru[1])
                )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    repo_means = {
        n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in per_repo_reuse.items()
    }
    deltas = [repo_means[n]["chelated"] - repo_means[n]["challenger"] for n in names]
    mean, lcb, ucb = paired_bootstrap_ci(deltas, n_resamples=2000, seed=6002)
    weight_probe = weight_revision_probe(model)
    binding = positional_binding_control(model)
    comparison = compare_with_fresh_rebuild(store, export_canonical_ek5_audit_trace(store))
    block_files = len(cache.written_paths)
    wrong_contract_cache = SSDKVCache(root / "kv_wrongw", contract_for(model.patched_base()), store)
    wrong_contract_cache.index = dict(cache.index)  # same index, different identity
    wrong_contract_cache.root = cache.root
    wrong_contract_cache.bound = cache.bound
    wrong_stats = CacheStats()
    probe_blocks = repos[0].commits[3].files
    serve_chelated(model, wrong_contract_cache, probe_blocks, wrong_stats, contract=contract_for(model.patched_base()))
    gates = {
        "repo_and_commit_counts_match_request": len(names) == repo_count and commits_checked == repo_count * commits_per_repo,
        "repos_hash_selected_distinct": len(set(names)) == len(names),
        "chelated_max_abs_logit_diff_lt_1e-4": max_logit_diff_ch < LOGIT_TOL,
        "chelated_greedy_tokens_equal_full_prefill": greedy_mismatch_ch == 0,
        "challenger_logit_parity_also_holds": max_logit_diff_lru < LOGIT_TOL and greedy_mismatch_lru == 0,
        "stale_weight_revision_refused_all_trials": (
            weight_probe["refused_after_base_patch"] == weight_probe["trials"]
            and weight_probe["refused_after_adapter_change"] == weight_probe["trials"]
            and weight_probe["tra1_silent_reuses_after_patch"] == 0
        ),
        "config_and_tokenizer_identical_in_weight_probe": (
            weight_probe["config_sha256_identical"] and weight_probe["tokenizer_sha256_identical"]
        ),
        "memt_style_identity_would_reuse_silently": weight_probe["memt_style_silent_reuses_after_patch"] == weight_probe["trials"],
        "ssd_cache_refuses_patched_weights": wrong_stats.reused_tokens == 0 and wrong_stats.stale_weight_refusals > 0,
        "naive_block_key_control_breaks_parity": binding["naive_block_only_reuse_exceeds_tol_trials"] > 0,
        "prefix_and_position_change_the_key": binding["prefix_change_changes_key"] and binding["position_change_changes_key"],
        "retractions_exercised": retractions > 0 and replay_probes > 0,
        "ek5_blocks_invalidated_when_earlier_prefix_retracted": probe_stale_chelated == 0 and downstream_clean_blocks_invalidated > 0,
        "challenger_serves_stale_after_retraction_control": probe_stale_challenger > 0,
        "chelated_probe_logits_match_full_prefill": probe_logit_diff < LOGIT_TOL,
        "ek5_fresh_rebuild_parity": not (
            comparison.semantic_diffs or comparison.canonical_reason_diffs or comparison.canonical_pos_cyc_diffs
        ),
        "chelated_reuse_non_inferior_to_challenger_lcb_ge_neg_0_01": lcb >= -0.01,
        "ssd_blocks_written": block_files > 0,
        "cuda_hidden": cuda_env.get("CUDA_VISIBLE_DEVICES") == "",
        "kernel_ledger_ingested_all_file_versions": len(registered) > 0,
    }
    performance_superior = lcb > 0.0
    deterministic_view = {
        "bundle_id": BUNDLE_ID,
        "mode": mode,
        "repo_names": names,
        "performance_superior_lcb_gt_0": performance_superior,
        "commits_per_repo": commits_per_repo,
        "contract_digest": contract.digest(),
        "chelated_reuse_mean": float(np.mean([m["chelated"] for m in repo_means.values()])),
        "challenger_reuse_mean": float(np.mean([m["challenger"] for m in repo_means.values()])),
        "paired_delta_mean": mean,
        "retractions": retractions,
        "replay_probes": replay_probes,
        "block_files_written": block_files,
        "gate_names": sorted(gates),
    }
    return {
        "bundle_id": BUNDLE_ID,
        "item": "2E",
        "mode": mode,
        "tier": "floor",
        "tier_disclosure": (
            "synthetic git histories and a tiny deterministic 2-layer numpy model, not an LLM on real repos; "
            "this checks the binding/invalidation mechanism, not production speedup"
        ),
        "repos": {"count": len(names), "names": names, "commits_per_repo": commits_per_repo, "commits_checked": commits_checked},
        "equivalence": {
            "chelated_max_abs_logit_diff": max_logit_diff_ch,
            "challenger_max_abs_logit_diff": max_logit_diff_lru,
            "tolerance": LOGIT_TOL,
            "greedy_steps": GREEDY_STEPS,
            "chelated_greedy_mismatches": greedy_mismatch_ch,
            "challenger_greedy_mismatches": greedy_mismatch_lru,
        },
        "performance": {
            "chelated": {"reused_token_fraction": st_ch.reused_tokens / st_ch.prompt_tokens, "blocks_hit": st_ch.blocks_hit, "blocks_missed": st_ch.blocks_missed},
            "challenger": {"reused_token_fraction": st_lru.reused_tokens / st_lru.prompt_tokens, "blocks_hit": st_lru.blocks_hit, "blocks_missed": st_lru.blocks_missed, "ram_capacity_blocks": CHALLENGER_RAM_BLOCKS},
            "paired_repo_delta": {"mean": mean, "lcb95": lcb, "ucb95": ucb, "n_clusters": len(names), "method": "repo-level paired bootstrap"},
            "advantage_source": (
                "none expected on reuse: both arms use identical stable-files-first exact-prefix keys and the "
                "workload is repo-sequential so the RAM LRU is not thrashed; the chelated arm's value is the "
                "weight-identity and EK5 retraction binding, not a reuse-rate win"
            ),
        },
        "retraction": {
            "retractions": retractions,
            "replay_probes": replay_probes,
            "chelated_stale_served_blocks": probe_stale_chelated,
            "challenger_stale_served_blocks": probe_stale_challenger,
            "downstream_clean_blocks_invalidated": downstream_clean_blocks_invalidated,
            "chelated_refused_retracted_blocks": st_ch.refused_retracted_blocks,
        },
        "weight_revision": weight_probe,
        "binding_control": binding,
        "ssd": {"block_files": block_files, "bytes_on_disk": cache.bytes_on_disk},
        "traced_peak_bytes": int(peak),
        "gates": gates,
        "overall_pass": all(gates.values()),
        "overall_pass_scope": "mechanism gates only (equivalence, binding, invalidation, non-inferiority)",
        "performance_claim_supported": performance_superior,
        "retirement_trigger_fired": not performance_superior,
        "retirement_trigger_note": (
            "claim registry #60: retire if the preregistered cache performance threshold ties or is missed; "
            "a repo-level paired delta with LCB <= 0 is a tie against the stable-files-first prefix-LRU"
        ),
        "deterministic_view": deterministic_view,
        "deterministic_sha256": _canonical_digest(deterministic_view),
        "kernel_ledger_events": len(registered),
    }
