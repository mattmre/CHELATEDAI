"""Post-bank drift-recovery conditions C5 / C5s / C5r (Phase II — H5b / S2b).

Three head-to-head conditions over the steering-post bank, all supervised by the
same held-out anchor (doc, drifted-query) pairs the single-adapter cycle uses:

- **C5  (living bank)** — build a bank of per-cluster correction posts, then each
  cycle anneal the lifecycle temperature, score post fitness, and prune+re-anneal
  low-fitness posts. The bank EVOLVES.
- **C5s (frozen SVF-style static bank)** — build the bank once, freeze it, re-apply
  every cycle. No lifecycle.
- **C5r (one-shot LD-MoLE-style router)** — build once, apply once (cycle 1 only).
  No lifecycle, no re-application.

The novel logic is `evolve_or_build_bank` — the build-once-vs-evolve + prune/
re-anneal orchestration — which is PURE (takes a state dict + an injected
`make_post` + plain anchor vectors; no engine), so it is fully stub-testable
without torch/GPU. `run_post_bank_cycle` is the thin engine wiring: trigger
(NDCG-drop) → evolve/build → apply-to-store. The real per-cluster bounded-adapter
trainer (`make_adapter_post_factory`) trains a standalone adapter per cluster.

Whether the living bank BEATS the static bank and the one-shot router is the
GPU-gated campaign's verdict and is NOT claimed here — this slice wires the three
conditions and proves the lifecycle is exercised + the store mutates.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from annealing_schedule import AnnealingSchedule
from post_bank_correction import MakePost
from post_bank_runtime import apply_post_bank_to_store, build_post_bank_from_anchor_pairs
from steering_post_bank import SteeringPostBank

POSTBANK_CONDITIONS = ("C5", "C5s", "C5r")


def evolve_or_build_bank(
    state: Optional[Dict[str, Any]],
    condition: str,
    anchor_pairs: Sequence[Mapping[str, Any]],
    vector_size: int,
    k: int,
    make_post: MakePost,
    schedule: AnnealingSchedule,
    cycle_index: int,
    seed: int,
    post_fitness: Optional[Mapping[str, float]] = None,
    prune_below: float = 0.5,
    min_posts: int = 1,
) -> Dict[str, Any]:
    """Build the bank (first call) or evolve it (C5 only). PURE — no engine.

    Returns ``{"bank", "state", "lifecycle"}``. ``state`` is threaded back by the
    caller (stashed on the engine) so the bank persists across cycles. ``lifecycle``
    records what the cycle did (built / annealed / pruned / re-annealed), which is
    the store-mutation-adjacent evidence that C5 is genuinely living.

    ``post_fitness`` (key -> score) is supplied by the caller from the previous
    apply (e.g. per-post hit counts); on a C5 evolve cycle it drives pruning. A
    re-annealed post is rebuilt from the SAME anchors (deterministic for a fixed
    seed), so re-anneal restores the pruned cluster's post identically.
    """
    if condition not in POSTBANK_CONDITIONS:
        raise ValueError(f"not a post-bank condition: {condition!r}")

    if state is None:
        bank = build_post_bank_from_anchor_pairs(
            anchor_pairs, vector_size, k, make_post, seed,
            prune_below=prune_below, min_posts=min_posts,
        )
        state = {
            "bank": bank,
            "built_cycle": cycle_index,
            "k": int(k),
            "seed": int(seed),
            "vector_size": int(vector_size),
        }
        return {"bank": bank, "state": state, "lifecycle": {"built": True, "n_posts": len(bank)}}

    bank: SteeringPostBank = state["bank"]
    lifecycle: Dict[str, Any] = {"built": False, "pruned": [], "reannealed": [], "temperature": None}

    # C5s (static) and C5r (one-shot) never evolve. C5r additionally does not
    # re-apply (the caller skips apply after the build cycle).
    if condition == "C5" and cycle_index > state["built_cycle"]:
        temperature = schedule.temperature(cycle_index - 1)
        bank.anneal_step(temperature)
        lifecycle["temperature"] = temperature
        if post_fitness:
            for key, score in post_fitness.items():
                if bank.post(key) is not None:
                    bank.record_fitness(key, float(score))

        # Re-anneal a pruned post by rebuilding it from the same anchors. The fresh
        # bank is deterministic (same seed/k/anchors), so post[key] is the original
        # centroid + a freshly-trained correction — built lazily, once per cycle.
        fresh_holder: Dict[str, SteeringPostBank] = {}

        def reanneal_factory(key: str):
            if "bank" not in fresh_holder:
                fresh_holder["bank"] = build_post_bank_from_anchor_pairs(
                    anchor_pairs, state["vector_size"], state["k"], make_post, state["seed"],
                    prune_below=prune_below, min_posts=min_posts,
                )
            post = fresh_holder["bank"].post(key)
            if post is None:
                raise KeyError(f"re-anneal: no post {key!r} in rebuilt bank")
            return post.centroid, post.correct

        result = bank.prune_and_reanneal(reanneal_factory, drift_fired=True)
        lifecycle["pruned"] = result["pruned"]
        lifecycle["reannealed"] = result["reannealed"]

    lifecycle["n_posts"] = len(bank)
    return {"bank": bank, "state": state, "lifecycle": lifecycle}


def make_adapter_post_factory(
    vector_size: int, bounded: bool, seed: int, steps: int = 30, learning_rate: float = 0.01,
    bound_epsilon: float = 0.01, max_correction: float = 0.5,
) -> MakePost:
    """Return a ``make_post`` that trains a standalone (optionally bounded) adapter
    per cluster on that cluster's (doc -> drifted-query) anchor pairs via InfoNCE,
    and wraps it as a numpy ``vec -> vec`` correction. Lazily imports torch.
    """

    def make_post(cluster_doc_vectors: np.ndarray, cluster_query_vectors: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        import torch

        from chelation_adapter import create_adapter
        from sedimentation_loss import SedimentationInfoNCELoss

        torch.manual_seed(int(seed))
        adapter = create_adapter(
            "mlp", input_dim=int(vector_size), bounded=bool(bounded),
            min_correction=float(bound_epsilon), max_correction=float(max_correction),
        )
        docs = torch.tensor(np.asarray(cluster_doc_vectors, dtype=np.float32), dtype=torch.float32)
        queries = torch.tensor(np.asarray(cluster_query_vectors, dtype=np.float32), dtype=torch.float32)
        if docs.shape[0] >= 1:
            loss_fn = SedimentationInfoNCELoss()
            optimizer = torch.optim.Adam(adapter.parameters(), lr=float(learning_rate))
            adapter.train()
            for _ in range(int(steps)):
                optimizer.zero_grad()
                loss = loss_fn(adapter(docs), queries)
                loss.backward()
                optimizer.step()
        adapter.eval()

        def correct(vec: np.ndarray) -> np.ndarray:
            arr = np.atleast_2d(np.asarray(vec, dtype=np.float32))
            with torch.no_grad():
                out = adapter(torch.tensor(arr, dtype=torch.float32)).detach().cpu().numpy()
            return out[0] if np.asarray(vec).ndim == 1 else out

        return correct

    return make_post


def run_post_bank_cycle(
    engine: Any,
    condition: str,
    config: Mapping[str, Any],
    drifted_eval_vectors: Optional[Mapping[str, np.ndarray]],
    eval_qrels: Optional[Mapping[str, Mapping[str, float]]],
    anchor_pairs: Optional[Sequence[Mapping[str, Any]]],
    baseline_ndcg: float,
    original_doc_points: Optional[Sequence[Mapping[str, Any]]],
    cycle_index: int,
    *,
    make_post_factory: Optional[MakePost] = None,
    schedule: Optional[AnnealingSchedule] = None,
) -> Dict[str, Any]:
    """Engine wiring for a post-bank condition cycle: NDCG-drop trigger → build/
    evolve the bank → apply to the store. The C5r one-shot router applies only on
    its build cycle. ``make_post_factory`` / ``schedule`` are injectable for tests.
    """
    if condition not in POSTBANK_CONDITIONS:
        raise ValueError(f"not a post-bank condition: {condition!r}")
    if drifted_eval_vectors is None or eval_qrels is None:
        raise ValueError(f"{condition} requires drifted eval vectors and eval qrels")

    from run_drift_recovery_experiment import evaluate_engine_with_query_vectors

    k_int = int(config.get("k", 10))
    current_ndcg, _ = evaluate_engine_with_query_vectors(
        engine, drifted_eval_vectors, eval_qrels, k_int
    )
    ndcg_drop = max(0.0, float(baseline_ndcg) - float(current_ndcg))
    trigger_threshold = float(config.get("trigger_threshold", 0.0))
    should_correct = ndcg_drop > trigger_threshold

    base = {
        "action": "post_bank_correction",
        "post_bank_kind": {"C5": "living", "C5s": "static", "C5r": "one_shot"}[condition],
        "detector_source": "ndcg_drop_vs_baseline",
        "should_correct": bool(should_correct),
        "pre_correction_ndcg": float(current_ndcg),
        "ndcg_drop": float(ndcg_drop),
        "correction_applied": False,
    }
    if not (should_correct and anchor_pairs):
        return base

    n_posts = int(config.get("post_bank_clusters", config.get("n_posts", 3)))
    seed = int(config.get("seed", 0))
    make_post = make_post_factory or make_adapter_post_factory(
        vector_size=int(engine.vector_size),
        bounded=(condition in {"C5", "C5s"}),
        seed=seed,
        steps=int(config.get("correction_steps", 30)),
        learning_rate=float(config.get("correction_lr", 0.01)),
        bound_epsilon=float(config.get("bound_epsilon", 0.01)),
    )
    sched = schedule or AnnealingSchedule(
        "cosine", n_cycles=int(config.get("cycles", 12)),
        t_start=float(config.get("max_temperature", 1.0)), t_end=0.0,
    )

    prior_state = getattr(engine, "_post_bank_state", None)
    # C5r applies only on its build cycle; after that it is a frozen no-op.
    if condition == "C5r" and prior_state is not None:
        return {**base, "correction_applied": False, "post_bank_kind": "one_shot",
                "note": "one_shot router already applied; no re-application"}

    # Per-post fitness for a C5 evolve cycle: docs served last apply (a proxy; the
    # campaign sweeps the fitness signal). Read before this cycle's apply.
    post_fitness: Optional[Dict[str, float]] = None
    if condition == "C5" and prior_state is not None:
        bank_prev: SteeringPostBank = prior_state["bank"]
        apply_logs = [e for e in bank_prev.lifecycle_log if e["action"] == "apply"]
        if apply_logs:
            post_fitness = {k: float(v) for k, v in apply_logs[-1]["per_post_hits"].items()}

    evolved = evolve_or_build_bank(
        prior_state, condition, anchor_pairs, int(engine.vector_size), n_posts,
        make_post, sched, cycle_index, seed,
        post_fitness=post_fitness,
        prune_below=float(config.get("post_prune_below", 0.5)),
        min_posts=int(config.get("post_min_posts", 1)),
    )
    engine._post_bank_state = evolved["state"]
    bank = evolved["bank"]

    applied = apply_post_bank_to_store(engine, bank, original_doc_points or [])
    return {
        **base,
        "correction_applied": bool(applied["correction_norm_stats"]["mean"] > 0.0),
        "n_posts": len(bank),
        "lifecycle": evolved["lifecycle"],
        "correction_norm_stats": applied["correction_norm_stats"],
        "per_post_hits": applied["per_post_hits"],
        "store_updated": applied["updated"],
    }
