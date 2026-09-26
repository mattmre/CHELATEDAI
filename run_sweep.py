import argparse
import itertools
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from antigravity_engine import AntigravityEngine
from benchmark_evolution import evaluate_ndcg, load_mteb_data
from chelation_adapter import create_adapter
from config import ChelationConfig
from sweep_corpus_restore import restore_collection, snapshot_collection


def remove_configured_adapter_weights():
    """Delete ChelationConfig.ADAPTER_WEIGHTS_PATH when that file exists."""
    weights_path = Path(ChelationConfig.ADAPTER_WEIGHTS_PATH)
    if weights_path.exists():
        weights_path.unlink()


def isolate_sweep_configuration(engine):
    """Drop one configuration's trained state. Keep the open Qdrant client.

    A second AntigravityEngine per grid row would open another client on the
    same path. The leak across rows is the adapter object, the chelation log,
    a trained projection, and the last evolution-strategy result.
    """
    remove_configured_adapter_weights()
    engine.adapter = create_adapter(
        adapter_type=ChelationConfig.ADAPTER_TYPE,
        input_dim=engine.vector_size,
        rank=ChelationConfig.LOW_RANK_ADAPTER_RANK,
    )
    log = getattr(engine, "chelation_log", None)
    if log is not None and hasattr(log, "clear"):
        log.clear()
    helper = getattr(engine, "teacher_helper", None)
    if helper is not None:
        if hasattr(helper, "begin_live_distillation"):
            helper.begin_live_distillation()
        if hasattr(helper, "_projection"):
            helper._projection = None
        projections = getattr(helper, "_projections", None)
        if isinstance(projections, dict):
            projections.clear()
    if hasattr(engine, "_last_es_result"):
        engine._last_es_result = None


def _snapshot_baseline_collection(client, collection_name):
    """Snapshot the baseline corpus. The sweep keeps this client open."""
    return snapshot_collection(client, collection_name)


def prepare_sweep_baseline(
    engine,
    corpus,
    queries,
    qrels,
    max_queries=None,
    snapshot_collection=None,
    batch_size=50,
):
    """Ingest a short corpus with raw vectors, then baseline a fresh adapter.

    Returns ``(base_score, corpus_snapshot)``. A failed ingest batch prints
    the error and returns ``1`` before the baseline score or the snapshot.
    """
    failure = _ingest_raw_corpus_if_short(engine, corpus, batch_size)
    if failure:
        return failure

    remove_configured_adapter_weights()
    fresh_adapter = create_adapter(
        adapter_type=ChelationConfig.ADAPTER_TYPE,
        input_dim=engine.vector_size,
        rank=ChelationConfig.LOW_RANK_ADAPTER_RANK,
    )
    engine.adapter = fresh_adapter
    base_score = evaluate_ndcg(engine, queries, qrels, max_queries=max_queries)
    print(f"Baseline NDCG@10: {base_score:.5f}")
    take_snapshot = _snapshot_baseline_collection if snapshot_collection is None else snapshot_collection
    corpus_snapshot = take_snapshot(engine.qdrant, engine.collection_name)
    return base_score, corpus_snapshot


def _ingest_raw_corpus_if_short(engine, corpus, batch_size):
    """Upsert the corpus with base vectors. Returns 1 on batch failure.

    A collection whose ``points_count`` is already at least the corpus
    length is still rewritten. Those stored vectors can be adapter
    outputs. Skipping them would publish that bake as the baseline.
    """
    info = engine.qdrant.get_collection(engine.collection_name)
    if info.points_count >= len(corpus):
        print(
            f"Collection already holds {info.points_count} points. "
            "Rewriting the corpus with raw vectors."
        )
    else:
        print(f"Collection empty. Ingesting {len(corpus)} documents...")
    from qdrant_client.models import PointStruct

    keys = list(corpus.keys())
    values = list(corpus.values())
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i + batch_size]
        batch_texts = values[i:i + batch_size]
        try:
            # embed_raw: EmbeddingBackend method for base vectors. It does not apply the adapter.
            embeddings = engine.embedding_backend.embed_raw(batch_texts)
            points = []
            for k, v, t in zip(batch_keys, embeddings, batch_texts):
                try:
                    pid = int(k)
                except (ValueError, TypeError):
                    pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(k)))
                points.append(PointStruct(id=pid, vector=v, payload={"text": t, "original_id": str(k)}))
            engine.qdrant.upsert(engine.collection_name, points)
            if i % 500 == 0:
                print(f"Ingested {i + len(batch_keys)}/{len(corpus)}")
        except Exception as e:
            print(f"Ingestion failed for batch {i}: {e}")
            return 1
    return 0


def run_parameter_sweep(task_name="SciFact", model_name="ollama:nomic-embed-text", output_file="sweep_results.json", max_queries=None, db_path=None):
    print(f"Starting parameter sweep on {task_name} using {model_name}")

    # Define the parameter grid
    learning_rates = [0.01, 0.1, 0.5]
    thresholds = [1, 2, 3]
    noise_scales = [0.0, 0.05, 0.2]  # 0.0 means disabled
    epochs_list = [5, 10, 20]

    # Load data once
    print(f"Loading MTEB data for {task_name}...")
    corpus, queries, qrels = load_mteb_data(task_name)
    if corpus is None:
        print("Failed to load data. Aborting.")
        return

    results = []

    # Generate all combinations
    combinations = list(itertools.product(learning_rates, thresholds, noise_scales, epochs_list))
    total_runs = len(combinations)

    print(f"Total configurations to test: {total_runs}")

    # Pre-calculate baseline so we don't have to do it every time
    print("Calculating baseline performance...")
    db_path = db_path or str(ChelationConfig.get_db_path(task_name))

    base_engine = AntigravityEngine(
        qdrant_location=str(db_path),
        model_name=model_name,
        chelation_p=85,
        use_quantization=True,
        use_centering=False
    )

    prepared = prepare_sweep_baseline(
        base_engine,
        corpus,
        queries,
        qrels,
        max_queries=max_queries,
        snapshot_collection=_snapshot_baseline_collection,
    )
    if isinstance(prepared, int):
        return prepared
    base_score, corpus_snapshot = prepared

    # Run the sweep
    for i, (lr, thresh, noise, epochs) in enumerate(combinations):
        print(f"[{i+1}/{total_runs}] Testing LR={lr}, Threshold={thresh}, Noise={noise}, Epochs={epochs}")

        # One Qdrant client stays open. Restore the baseline corpus before
        # this configuration's adapter reset and sedimentation upsert.
        engine = base_engine
        restore_collection(engine.qdrant, engine.collection_name, corpus_snapshot)
        isolate_sweep_configuration(engine)
        evaluate_ndcg(engine, queries, qrels, max_queries=max_queries)

        # Enable noise injection temporarily via config patching
        original_noise_enabled = ChelationConfig.NOISE_INJECTION_ENABLED
        original_noise_scale = ChelationConfig.NOISE_INJECTION_BASE_SCALE

        if noise > 0:
            ChelationConfig.NOISE_INJECTION_ENABLED = True
            ChelationConfig.NOISE_INJECTION_BASE_SCALE = noise
        else:
            ChelationConfig.NOISE_INJECTION_ENABLED = False

        # Run sedimentation
        sediment_start = time.time()
        engine.run_sedimentation_cycle(threshold=thresh, learning_rate=lr, epochs=epochs, noise_injection=noise if noise > 0 else None)
        sediment_time = time.time() - sediment_start

        # Restore config
        ChelationConfig.NOISE_INJECTION_ENABLED = original_noise_enabled
        ChelationConfig.NOISE_INJECTION_BASE_SCALE = original_noise_scale

        # Evaluate post-learning
        post_score = evaluate_ndcg(engine, queries, qrels, max_queries=max_queries)
        gain = post_score - base_score

        print(f"Post-Learning NDCG@10: {post_score:.5f} (Gain: {gain:+.5f})")

        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "learning_rate": lr,
                "threshold": thresh,
                "noise_scale": noise,
                "epochs": epochs
            },
            "metrics": {
                "baseline_ndcg": base_score,
                "post_ndcg": post_score,
                "gain": gain,
                "sediment_time": sediment_time
            }
        }
        results.append(result_entry)

        # Save incrementally
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Reset adapter weights so the next run starts fresh
        remove_configured_adapter_weights()

    print(f"Sweep completed. Results saved to {output_file}")

    # Find best
    best_run = max(results, key=lambda x: x["metrics"]["gain"])
    print("Best Configuration:")
    print(json.dumps(best_run, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Sweep for Chelation Sedimentation")
    parser.add_argument("--task", type=str, default="SciFact", help="MTEB Task")
    parser.add_argument("--model", type=str, default="ollama:nomic-embed-text", help="Embedding Model")
    parser.add_argument("--out", type=str, default="sweep_results.json", help="Output JSON file")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query cap for faster sweep iteration")
    parser.add_argument("--db-path", type=str, default=None, help="Optional isolated Qdrant path for this sweep")

    args = parser.parse_args()
    exit_code = run_parameter_sweep(args.task, args.model, args.out, max_queries=args.max_queries, db_path=args.db_path)
    if exit_code:
        sys.exit(exit_code)
