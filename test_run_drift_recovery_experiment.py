from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from chelation_adapter import BoundedAdapter
from run_drift_recovery_experiment import DriftRecoveryConfig, CONDITIONS, run_experiment


class TinyEmbeddingBackend:
    vector_size = 4

    def __init__(self):
        self.vectors = {
            "alpha topic": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "beta topic": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "gamma topic": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "delta topic": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }

    def embed_raw(self, texts):
        rows = []
        for text in texts:
            vector = self.vectors.get(str(text), np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32))
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32)


class StubSwapBackend:
    """A *different* frozen encoder used by the query-encoder-swap arena.

    It maps each topic to a CYCLICALLY PERMUTED one-hot relative to the original
    TinyEmbeddingBackend (alpha->dim1, beta->dim2, gamma->dim3, delta->dim0).
    After the near-identity seeded projection, a drifted query for "alpha topic"
    lands near the basis dimension the ORIGINAL store assigned to "beta topic",
    so it retrieves the wrong doc against the original store (C0/C2 stay low) but
    matches a doc re-embedded with this same swap encoder (C2O recovers).
    """

    vector_size = 4

    def __init__(self):
        self.vectors = {
            "alpha topic": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "beta topic": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "gamma topic": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "delta topic": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

    def embed_raw(self, texts):
        rows = []
        for text in texts:
            vector = self.vectors.get(str(text), np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32))
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32)


class TestRunDriftRecoveryExperiment(unittest.TestCase):
    def setUp(self):
        # Order-independence isolation against torch GLOBAL state a seed does not
        # control. run_experiment() reseeds random/numpy/torch internally, so the
        # seed is not the issue; but the trajectory floats come from a torch
        # matmul whose value/executability depends on torch globals an earlier
        # test in a full-suite run can leak:
        #   - default dtype: a leaked torch.float64 makes adapter init float64 and
        #     CRASHES the matmul (Float vs Double) — pinned to float32.
        #   - default device: a leaked 'cuda' pushes the matmul onto the GPU and
        #     crashes the .numpy() readback (and would diverge low bits) — pinned
        #     to 'cpu' (guarded; the API exists on torch>=2.0).
        #   - intra-op thread count: defensively pinned to 1 for a deterministic
        #     single reduction order.
        # The golden comparison itself is also tolerance-based (see
        # _assert_trajectory_close) so residual low-bit FP differences from any
        # un-enumerated global cannot flake it. State is restored via addCleanup
        # so this test neither depends on nor pollutes sibling tests.
        self._prev_num_threads = torch.get_num_threads()
        self._prev_default_dtype = torch.get_default_dtype()
        self._prev_default_device = None
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        if hasattr(torch, "set_default_device"):
            getter = getattr(torch, "get_default_device", None)
            self._prev_default_device = getter() if getter is not None else torch.device("cpu")
            torch.set_default_device("cpu")
        np.random.seed(0)
        torch.manual_seed(0)
        self.addCleanup(self._restore_torch_state)

        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.corpus = {
            "d-alpha": "alpha topic",
            "d-beta": "beta topic",
            "d-gamma": "gamma topic",
            "d-delta": "delta topic",
        }
        self.queries = {
            "q-alpha": "alpha topic",
            "q-beta": "beta topic",
        }
        self.qrels = {
            "q-alpha": {"d-alpha": 1.0},
            "q-beta": {"d-beta": 1.0},
        }

    def test_all_conditions_write_well_formed_json_through_engine_path(self):
        with self._patched_backend():
            for condition in CONDITIONS:
                # C2O/C3a/C4a and the post-bank conditions C5/C5s/C5r are
                # query-encoder-swap-only conditions and are not valid for the
                # rotation drift this case exercises; they are covered separately
                # by the query_encoder_swap arena tests below.
                if condition in {"C2O", "C3a", "C4a", "C5", "C5s", "C5r"}:
                    continue
                output = Path(self.tempdir.name) / f"{condition}.json"
                config = self._config(condition, output)
                result = run_experiment(config, self.corpus, self.queries, self.qrels)

                self.assertEqual(result["record_type"], "drift_recovery_experiment")
                self.assertEqual(result["config"]["condition"], condition)
                self.assertEqual(result["config"]["seed"], 123)
                self.assertEqual(result["config"]["injection_index"], 0)
                self.assertEqual(result["drift_manifest"]["injection_index"], 0)
                self.assertEqual(len(result["recovery"]["trajectory"]), 2)
                self.assertIn("baseline_ndcg", result["recovery"])
                self.assertTrue(output.exists())

                loaded = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(loaded["config"]["condition"], condition)
                self.assertEqual(loaded["drift_manifest"]["seed"], 123)
                self.assertEqual(loaded["cycle_errors"], [])
                if condition == "C2":
                    first_meta = loaded["recovery"]["trajectory"][0]["metadata"]
                    self.assertEqual(first_meta["action"], "maintenance_reindex_affected")
                    self.assertEqual(first_meta["refresh"]["updated"], loaded["drift_manifest"]["affected_count"])
                if condition in {"C3", "C4"}:
                    self.assertGreater(loaded["correction_norm_stats"]["count"], 0)
                    first_meta = loaded["recovery"]["trajectory"][0]["metadata"]
                    self.assertEqual(
                        first_meta["detector_source"],
                        "AntigravityEngine._compute_annealing_drift_magnitude",
                    )
                    self.assertIn("drift_magnitude", first_meta["annealing_observation"])
                    self.assertIn("should_correct", first_meta)
                    self.assertIn("sedimentation_attempted", first_meta)
                    self.assertIn("correction_applied", first_meta)
                    self.assertNotIn("corrected", first_meta)

    def test_c3_uses_bounded_adapter_and_c4_uses_unbounded_adapter(self):
        observed_adapter_types = {}

        def capture_cycle(engine, condition, queries, drift_manifest, run_config=None, query_drift=None, **kwargs):
            observed_adapter_types[condition] = isinstance(engine.adapter, BoundedAdapter)
            return {"action": "captured"}

        with self._patched_backend(), patch(
            "run_drift_recovery_experiment._run_condition_cycle",
            side_effect=capture_cycle,
        ):
            run_experiment(self._config("C3", Path(self.tempdir.name) / "c3.json"), self.corpus, self.queries, self.qrels)
            run_experiment(self._config("C4", Path(self.tempdir.name) / "c4.json"), self.corpus, self.queries, self.qrels)

        self.assertTrue(observed_adapter_types["C3"])
        self.assertFalse(observed_adapter_types["C4"])

    def test_same_seed_twice_produces_identical_trajectory(self):
        with self._patched_backend():
            first = run_experiment(
                self._config("C3", Path(self.tempdir.name) / "first.json"),
                self.corpus,
                self.queries,
                self.qrels,
            )
            second = run_experiment(
                self._config("C3", Path(self.tempdir.name) / "second.json"),
                self.corpus,
                self.queries,
                self.qrels,
            )

        self.assertEqual(first["drift_manifest"], second["drift_manifest"])
        self.assertEqual(first["recovery"]["trajectory"], second["recovery"]["trajectory"])

    def test_explicit_default_knobs_preserve_default_trajectory(self):
        with self._patched_backend():
            implicit = run_experiment(
                self._config("C3", Path(self.tempdir.name) / "implicit.json"),
                self.corpus,
                self.queries,
                self.qrels,
            )
            explicit_config = self._config("C3", Path(self.tempdir.name) / "explicit.json")
            explicit_config = DriftRecoveryConfig(
                **{
                    **explicit_config.__dict__,
                    "bound_epsilon": 0.01,
                    "trigger_threshold": 0.0,
                    "max_temperature": 1.0,
                    "epochs_scale": 1.0,
                }
            )
            explicit = run_experiment(explicit_config, self.corpus, self.queries, self.qrels)

        # Exact: explicit-default knobs must reproduce the implicit-default
        # trajectory bit-for-bit (same process/environment -> robust to globals).
        self.assertEqual(implicit["recovery"]["trajectory"], explicit["recovery"]["trajectory"])
        # Tolerance on float leaves only: guards against trajectory regressions
        # without flaking on ~1e-16 FP noise from un-enumerated torch globals.
        self._assert_trajectory_close(
            implicit["recovery"]["trajectory"], self._pre_change_c3_default_golden_trajectory()
        )
        first_meta = implicit["recovery"]["trajectory"][0]["metadata"]
        self.assertNotIn("knobs", first_meta)
        self.assertNotIn("epochs_scale", first_meta.get("annealing_settings", {}))

    def test_c3_records_non_default_knobs_and_applies_epoch_scale(self):
        config = self._config("C3", Path(self.tempdir.name) / "knobs.json")
        config = DriftRecoveryConfig(
            **{
                **config.__dict__,
                "bound_epsilon": 0.05,
                "trigger_threshold": 0.0,
                "max_temperature": 0.5,
                "epochs_scale": 2.0,
            }
        )

        with self._patched_backend():
            result = run_experiment(config, self.corpus, self.queries, self.qrels)

        first_meta = result["recovery"]["trajectory"][0]["metadata"]
        self.assertEqual(
            first_meta["knobs"],
            {
                "bound_epsilon": 0.05,
                "trigger_threshold": 0.0,
                "max_temperature": 0.5,
                "epochs_scale": 2.0,
            },
        )
        self.assertEqual(first_meta["annealing_settings"]["epochs_scale"], 2.0)
        self.assertGreaterEqual(first_meta["annealing_settings"]["effective_epochs"], 2)
        self.assertGreater(result["correction_norm_stats"]["mean"], 0.03)

    def test_invalid_knob_values_raise(self):
        bad_config = self._config("C3", Path(self.tempdir.name) / "bad.json")
        bad_config = DriftRecoveryConfig(**{**bad_config.__dict__, "bound_epsilon": 0.0})

        with self.assertRaisesRegex(ValueError, "bound_epsilon"):
            run_experiment(bad_config, self.corpus, self.queries, self.qrels)

    def _restore_torch_state(self):
        # Restore (via addCleanup) the global torch knobs we pinned in setUp so
        # this test does not leak single-threaded / float32 / cpu state onto
        # later tests.
        torch.set_num_threads(self._prev_num_threads)
        torch.set_default_dtype(self._prev_default_dtype)
        if self._prev_default_device is not None and hasattr(torch, "set_default_device"):
            torch.set_default_device(self._prev_default_device)

    def _assert_trajectory_close(self, actual, expected, path="trajectory"):
        """Compare two trajectory structures: float leaves within a tight
        tolerance (robust to ~1e-16 FP-environment noise), everything else
        (ints, bools, strings, list lengths, dict keys) exact. Keeps the test a
        real regression guard while immune to un-enumerated FP-affecting globals.
        """
        if isinstance(expected, bool) or isinstance(actual, bool):
            self.assertEqual(actual, expected, msg=path)
        elif isinstance(expected, float) or isinstance(actual, float):
            self.assertIsInstance(actual, (int, float), msg=path)
            self.assertTrue(
                np.isclose(float(actual), float(expected), rtol=1e-9, atol=1e-12),
                msg=f"{path}: {actual!r} != {expected!r} (beyond tolerance)",
            )
        elif isinstance(expected, dict):
            self.assertIsInstance(actual, dict, msg=path)
            self.assertEqual(set(actual.keys()), set(expected.keys()), msg=path)
            for key in expected:
                self._assert_trajectory_close(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected), msg=path)
            for i, (a, e) in enumerate(zip(actual, expected)):
                self._assert_trajectory_close(a, e, f"{path}[{i}]")
        else:
            self.assertEqual(actual, expected, msg=path)

    def _config(self, condition, output):
        return DriftRecoveryConfig(
            task="Tiny",
            condition=condition,
            drift="rotation",
            fraction=0.5,
            angle=25.0,
            sigma=0.05,
            cycles=2,
            seed=123,
            max_queries=2,
            sample_docs=4,
            output=str(output),
            model="tiny-local",
            device="cpu",
        )

    # --- query_encoder_swap arena -------------------------------------------------

    def _swap_config(self, condition, output, cycles=1):
        return DriftRecoveryConfig(
            task="Tiny",
            condition=condition,
            drift="query_encoder_swap",
            fraction=0.5,
            angle=25.0,
            sigma=0.05,
            cycles=cycles,
            seed=123,
            max_queries=4,
            sample_docs=4,
            output=str(output),
            model="tiny-local",
            device="cpu",
            swap_model="stub-swap",
        )

    def _patched_swap_backend(self):
        @contextmanager
        def manager():
            with ExitStack() as stack:
                stack.enter_context(
                    patch("antigravity_engine.create_embedding_backend", return_value=TinyEmbeddingBackend())
                )
                stack.enter_context(patch("antigravity_engine.get_logger", return_value=MagicMock()))
                # The swap encoder is resolved lazily inside QueryEncoderDrift via
                # embedding_backend.create_embedding_backend(swap_model_name); patch
                # it so the arena runs without the real mpnet model.
                stack.enter_context(
                    patch("embedding_backend.create_embedding_backend", return_value=StubSwapBackend())
                )
                yield

        return manager()

    def _swap_corpus_queries_qrels(self):
        corpus = {
            "d-alpha": "alpha topic",
            "d-beta": "beta topic",
            "d-gamma": "gamma topic",
            "d-delta": "delta topic",
        }
        queries = {
            "q-alpha": "alpha topic",
            "q-beta": "beta topic",
            "q-gamma": "gamma topic",
            "q-delta": "delta topic",
        }
        qrels = {
            "q-alpha": {"d-alpha": 1.0},
            "q-beta": {"d-beta": 1.0},
            "q-gamma": {"d-gamma": 1.0},
            "q-delta": {"d-delta": 1.0},
        }
        return corpus, queries, qrels

    def test_query_encoder_swap_oracle_breaker_c2_noop_c2o_recovers(self):
        """Load-bearing harness-level oracle-breaker.

        Baseline (native queries vs original store) is high. After the
        query-encoder upgrade, C0 (no correction) drops. C2 (re-embed docs with
        the ORIGINAL model) is a no-op and stays low. C2O (re-embed docs with the
        SWAP model + the same frozen projection) recovers materially above C2.
        """
        corpus, queries, qrels = self._swap_corpus_queries_qrels()

        finals = {}
        baselines = {}
        with self._patched_swap_backend():
            for condition in ("C0", "C2", "C2O"):
                output = Path(self.tempdir.name) / f"swap-{condition}.json"
                result = run_experiment(self._swap_config(condition, output), corpus, queries, qrels)
                baselines[condition] = result["baseline"]["ndcg_at_10"]
                finals[condition] = result["recovery"]["trajectory"][-1]["ndcg"]
                self.assertEqual(result["drift_manifest"]["drift"], "query_encoder_swap")

        # All conditions share the same baseline (pre-drift, native queries) and
        # it retrieves perfectly.
        for condition in ("C0", "C2", "C2O"):
            self.assertEqual(baselines[condition], 1.0, msg=f"baseline {condition}")
        # Drift degrades retrieval (C0).
        self.assertLess(finals["C0"], 0.5)
        # C2 re-embed-with-original is a proven no-op for query-side drift.
        self.assertEqual(finals["C2"], finals["C0"])
        self.assertLess(finals["C2"], 0.5)
        # C2O recovers fully and is materially above C2.
        self.assertEqual(finals["C2O"], 1.0)
        self.assertGreater(finals["C2O"], finals["C2"] + 0.5)

    def test_query_encoder_swap_c2_action_reembeds_with_original_model(self):
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            result = run_experiment(
                self._swap_config("C2", Path(self.tempdir.name) / "swap-c2-action.json"),
                corpus,
                queries,
                qrels,
            )
        meta = result["recovery"]["trajectory"][0]["metadata"]
        self.assertEqual(meta["action"], "maintenance_reembed_original_model")
        self.assertEqual(meta["refresh"]["updated"], len(corpus))

    def test_query_encoder_swap_c2o_action_reembeds_with_swap_model(self):
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            result = run_experiment(
                self._swap_config("C2O", Path(self.tempdir.name) / "swap-c2o-action.json"),
                corpus,
                queries,
                qrels,
            )
        meta = result["recovery"]["trajectory"][0]["metadata"]
        self.assertEqual(meta["action"], "oracle_reembed_swap_model")
        self.assertEqual(meta["refresh"]["updated"], len(corpus))

    def test_query_encoder_swap_manifest_records_projection_checksum(self):
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            result = run_experiment(
                self._swap_config("C0", Path(self.tempdir.name) / "swap-manifest.json"),
                corpus,
                queries,
                qrels,
            )
        manifest = result["drift_manifest"]
        self.assertEqual(manifest["drift"], "query_encoder_swap")
        self.assertEqual(manifest["swap_model"], "stub-swap")
        self.assertEqual(manifest["store_dim"], 4)
        self.assertEqual(manifest["swap_dim"], 4)
        self.assertEqual(manifest["injection_index"], 0)
        self.assertEqual(len(manifest["projection_checksum"]), 64)
        self.assertEqual(result["config"]["injection_index"], 0)

    def test_query_encoder_swap_same_seed_produces_identical_trajectory(self):
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            first = run_experiment(
                self._swap_config("C2O", Path(self.tempdir.name) / "swap-det-1.json"),
                corpus,
                queries,
                qrels,
            )
            second = run_experiment(
                self._swap_config("C2O", Path(self.tempdir.name) / "swap-det-2.json"),
                corpus,
                queries,
                qrels,
            )
        self.assertEqual(first["drift_manifest"], second["drift_manifest"])
        self.assertEqual(first["recovery"]["trajectory"], second["recovery"]["trajectory"])

    def test_c2o_requires_query_encoder_swap_drift(self):
        with self._patched_swap_backend():
            bad = self._config("C2O", Path(self.tempdir.name) / "bad-c2o.json")
            with self.assertRaisesRegex(ValueError, "C2O"):
                run_experiment(bad, self.corpus, self.queries, self.qrels)

    # --- supervised closed loop (C3a / C4a, PR-A2b) -------------------------------

    def _supervised_swap_config(self, condition, output, anchor_fraction=0.5, cycles=1):
        config = self._swap_config(condition, output, cycles=cycles)
        return DriftRecoveryConfig(**{**config.__dict__, "anchor_fraction": anchor_fraction})

    def test_c3a_actuator_fires_and_writes_to_store(self):
        """LOAD-BEARING: the v1 closed loop NEVER fired (correction_applied=0/36,
        store byte-identical). This proves PR-A2b's supervised actuator FIRES and
        WRITES — should_correct True, correction_applied True (store checksum
        changed), and the mean correction norm is well above the bound floor.
        """
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            result = run_experiment(
                self._supervised_swap_config("C3a", Path(self.tempdir.name) / "c3a-fires.json"),
                corpus,
                queries,
                qrels,
            )
        meta = result["recovery"]["trajectory"][-1]["metadata"]
        self.assertEqual(meta["action"], "supervised_anchor_infonce_correction")
        self.assertTrue(meta["should_correct"], "trigger must cross threshold")
        self.assertTrue(meta["sedimentation_attempted"])
        # The actuator actually mutated the store (the v1 failure was this == False).
        self.assertTrue(meta["correction_applied"], "store checksum must change")
        self.assertGreater(meta["anchor_count"], 0)
        self.assertGreater(meta["ndcg_drop"], 0.0)
        norm_stats = meta["correction_norm_stats"]
        self.assertGreater(norm_stats["count"], 0)
        # Mean correction is well above the BoundedAdapter floor (bound_epsilon=0.01).
        self.assertGreater(norm_stats["mean"], 0.01)
        # And the run-level aggregate reflects a real, non-trivial correction.
        self.assertGreater(result["correction_norm_stats"]["mean"], 0.01)

    def test_post_bank_conditions_fire_and_mutate_store_in_swap_arena(self):
        """End-to-end (H5b/S2b): C5/C5s/C5r build a per-cluster post-bank, fire
        (should_correct), and MUTATE the store (correction_applied) through the real
        run_experiment path in the query-encoder-swap arena. The second block forces
        C5's prune/re-anneal lifecycle to fire end-to-end. Whether C5 BEATS C5s/C5r
        is the GPU campaign's verdict — NOT asserted here."""
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        kinds = {"C5": "living", "C5s": "static", "C5r": "one_shot"}
        with self._patched_swap_backend():
            for cond in ("C5", "C5s", "C5r"):
                result = run_experiment(
                    self._supervised_swap_config(cond, Path(self.tempdir.name) / f"{cond}-pb.json", cycles=2),
                    corpus, queries, qrels,
                )
                build = result["recovery"]["trajectory"][0]["metadata"]
                self.assertEqual(build["action"], "post_bank_correction")
                self.assertEqual(build["post_bank_kind"], kinds[cond])
                self.assertTrue(build["should_correct"], f"{cond} trigger must fire")
                self.assertTrue(build["correction_applied"], f"{cond} must mutate the store")
                self.assertGreater(build["n_posts"], 0)
                self.assertGreater(build["correction_norm_stats"]["mean"], 0.0)
                self.assertTrue(build["lifecycle"]["built"])

            # C5 living: FORCE the prune/re-anneal lifecycle end-to-end on the evolve
            # cycle. prune_below above any post's fitness means every post but the
            # min_posts fittest is a prune candidate; with cycles=2 the final cycle's
            # cosine temperature is 0, so the effective threshold == post_prune_below.
            # With 2 clusters + min_posts 1, cycle 2 prunes 1 post and re-anneals it
            # from the same anchors -> bank restored to 2 (net no shrink). This
            # distinguishes the LIVING bank from the static/one-shot banks, which
            # never prune/re-anneal.
            base = self._supervised_swap_config("C5", Path(self.tempdir.name) / "c5-prune.json", cycles=2)
            forced = DriftRecoveryConfig(**{**base.__dict__, "post_bank_clusters": 2,
                                            "post_prune_below": 1e9, "post_min_posts": 1})
            traj = run_experiment(forced, corpus, queries, qrels)["recovery"]["trajectory"]
            self.assertEqual(len(traj), 2)
            cycle2 = traj[1]["metadata"]
            self.assertTrue(cycle2["should_correct"], "C5 cycle 2 must re-fire")
            self.assertTrue(cycle2["correction_applied"])
            self.assertFalse(cycle2["lifecycle"]["built"])           # evolve, not rebuild
            self.assertEqual(cycle2["lifecycle"]["temperature"], 0.0)
            self.assertTrue(cycle2["lifecycle"]["pruned"], "prune must fire end-to-end")
            self.assertTrue(cycle2["lifecycle"]["reannealed"], "re-anneal must restore the post")
            self.assertEqual(cycle2["n_posts"], 2)                   # net no shrink

    def test_c4a_unbounded_actuator_fires_and_writes(self):
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            result = run_experiment(
                self._supervised_swap_config("C4a", Path(self.tempdir.name) / "c4a-fires.json"),
                corpus,
                queries,
                qrels,
            )
        meta = result["recovery"]["trajectory"][-1]["metadata"]
        self.assertFalse(meta["bounded"])
        self.assertTrue(meta["should_correct"])
        self.assertTrue(meta["correction_applied"])
        # Unbounded correction is free to be larger than the bounded C3a floor.
        self.assertGreater(meta["correction_norm_stats"]["mean"], 0.01)

    def test_c3a_fires_writes_and_records_honest_eval_outcome(self):
        """C3a fires + writes, but on the cyclic-permutation stub geometry a
        bounded MLP trained on sparse held-out anchors does NOT generalize the
        per-query realignment to the disjoint eval queries — so it does not beat
        the no-correction baseline here. This is the HONEST finding (the
        documented bounded-adapter / sparse-anchor capacity limit), and the
        oracle C2O remains the only full recovery. Asserted as-is, not tuned.

        APPLES-TO-APPLES: the C0 baseline is run at the SAME anchor_fraction as
        C3a (0.5), so the anchor/eval split is identical and both score the SAME
        disjoint eval subset. (Comparing C3a@af=0.5 against C0@af=0 would measure
        different query subsets — a different bug Tier B flagged; this avoids it.)
        """
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        finals = {}
        with self._patched_swap_backend():
            # C0 at af=0.5: no correction, but the split is active so it scores
            # the identical eval subset C3a is scored on (fair baseline).
            r_c0 = run_experiment(
                self._supervised_swap_config("C0", Path(self.tempdir.name) / "honest-c0.json", anchor_fraction=0.5),
                corpus,
                queries,
                qrels,
            )
            finals["C0_fair"] = r_c0["recovery"]["trajectory"][-1]["ndcg"]
            c0_eval_ids = set(r_c0["anchor_eval_split"]["eval_ids"])
            # C2O oracle at af=0 (re-embeds all docs into the swap space — full recovery).
            r_c2o = run_experiment(
                self._swap_config("C2O", Path(self.tempdir.name) / "honest-c2o.json"),
                corpus,
                queries,
                qrels,
            )
            finals["C2O"] = r_c2o["recovery"]["trajectory"][-1]["ndcg"]
            r_c3a = run_experiment(
                self._supervised_swap_config("C3a", Path(self.tempdir.name) / "honest-c3a.json"),
                corpus,
                queries,
                qrels,
            )
        meta = r_c3a["recovery"]["trajectory"][-1]["metadata"]
        finals["C3a"] = r_c3a["recovery"]["trajectory"][-1]["ndcg"]
        c3a_eval_ids = set(r_c3a["anchor_eval_split"]["eval_ids"])
        # Same seed -> identical anchor/eval split, so C0_fair and C3a score the
        # SAME eval subset (this is what makes the comparison apples-to-apples).
        self.assertEqual(c0_eval_ids, c3a_eval_ids)
        # The loop fired and wrote (the load-bearing point).
        self.assertTrue(meta["correction_applied"])
        # Honest outcome: against the fair same-subset baseline, the supervised
        # correction does not beat no-correction on this geometry (it does not
        # generalize from the sparse anchors), and never exceeds the oracle.
        self.assertLessEqual(finals["C3a"], finals["C2O"])
        self.assertLessEqual(finals["C3a"], finals["C0_fair"] + 1e-9)
        # The oracle still recovers fully — proves the arena itself is recoverable.
        self.assertEqual(finals["C2O"], 1.0)

    def test_c3a_multi_cycle_is_idempotent(self):
        """With per-cycle adapter re-init from the fixed original-doc snapshot,
        each supervised cycle is a pure function of (snapshot, anchors, seed), so
        a multi-cycle run produces an identical correction and NDCG every cycle —
        no drift/compounding (Tier B caught the pre-fix accumulating-weights drift).
        """
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            result = run_experiment(
                self._supervised_swap_config("C3a", Path(self.tempdir.name) / "c3a-multicycle.json", cycles=3),
                corpus,
                queries,
                qrels,
            )
        traj = result["recovery"]["trajectory"]
        self.assertEqual(len(traj), 3)
        ndcgs = [cycle["ndcg"] for cycle in traj]
        norms = [cycle["metadata"]["correction_norm_stats"]["mean"] for cycle in traj]
        # Flat trajectory: every cycle's NDCG and correction magnitude identical.
        self.assertEqual(len(set(ndcgs)), 1, f"NDCG drifted across cycles: {ndcgs}")
        for norm in norms[1:]:
            self.assertAlmostEqual(norm, norms[0], places=6, msg=f"norm drifted: {norms}")
        for cycle in traj:
            self.assertTrue(cycle["metadata"]["correction_applied"])

    def test_c3a_anchor_eval_split_is_disjoint_and_recorded(self):
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            result = run_experiment(
                self._supervised_swap_config("C3a", Path(self.tempdir.name) / "c3a-split.json"),
                corpus,
                queries,
                qrels,
            )
        split = result["anchor_eval_split"]
        self.assertTrue(split["active"])
        self.assertGreater(split["anchor_count"], 0)
        self.assertGreater(split["eval_count"], 0)
        self.assertEqual(split["anchor_count"] + split["eval_count"], len(queries))
        self.assertTrue(set(split["anchor_ids"]).isdisjoint(set(split["eval_ids"])))
        self.assertEqual(result["config"]["anchor_count"], split["anchor_count"])
        self.assertEqual(result["config"]["eval_count"], split["eval_count"])
        # Eval NDCG is measured on the eval subset only.
        meta = result["recovery"]["trajectory"][-1]["metadata"]
        self.assertEqual(meta["evaluated_queries"], split["eval_count"])

    def test_c3a_same_seed_produces_identical_trajectory(self):
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            first = run_experiment(
                self._supervised_swap_config("C3a", Path(self.tempdir.name) / "c3a-det-1.json"),
                corpus,
                queries,
                qrels,
            )
            second = run_experiment(
                self._supervised_swap_config("C3a", Path(self.tempdir.name) / "c3a-det-2.json"),
                corpus,
                queries,
                qrels,
            )
        self.assertEqual(first["drift_manifest"], second["drift_manifest"])
        self.assertEqual(first["recovery"]["trajectory"], second["recovery"]["trajectory"])
        self.assertEqual(first["anchor_eval_split"], second["anchor_eval_split"])

    def test_supervised_conditions_have_no_build_time_bounded_adapter(self):
        """Regression (H1): the bounded adapter must NOT be live at ingest/baseline.

        ``embed()`` applies engine.adapter to every ingested doc vector and to the
        baseline query path. A build-time *bounded* adapter (the old behaviour for
        C3a) baked its ~bound_epsilon correction FLOOR into the stored vectors and
        the pre-correction baseline, making C3a's baseline diverge from
        C0/C2/C2O/C4a on dense-relevance data (NFCorpus) and breaking the
        "all conditions scored on the same eval subset" invariant. C3a/C4a
        re-create the adapter per correction cycle, so at build/baseline time
        neither must carry a bounded adapter.
        """
        observed_build_time = {}

        def capture_cycle(engine, condition, queries, drift_manifest, run_config=None, **kwargs):
            # Adapter state as the cycle is entered == post-build, pre-correction.
            observed_build_time[condition] = isinstance(engine.adapter, BoundedAdapter)
            return {"action": "captured"}

        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend(), patch(
            "run_drift_recovery_experiment._run_condition_cycle",
            side_effect=capture_cycle,
        ):
            for cond in ("C3a", "C4a"):
                run_experiment(
                    self._supervised_swap_config(cond, Path(self.tempdir.name) / f"{cond}-build.json"),
                    corpus,
                    queries,
                    qrels,
                )
        self.assertFalse(
            observed_build_time["C3a"], "C3a must not carry a build-time bounded adapter (H1)"
        )
        self.assertFalse(observed_build_time["C4a"])

    def test_supervised_correction_is_bounded_for_c3a_unbounded_for_c4a(self):
        """The bounded/unbounded property is enforced on the LIVE adapter object
        inside the real correction cycle (engine.adapter is re-created per cycle).

        We spy on the ACTUAL adapter at correction-apply time rather than reading
        ``meta['bounded']`` — that field is just ``condition == 'C3a'`` echoed back,
        so asserting it would be tautological and would not catch a regression in
        the per-cycle ``bounded=`` wiring.
        """
        import run_drift_recovery_experiment as mod

        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        observed = {}
        current = {"cond": None}
        original_apply = mod._apply_adapter_to_all_docs

        def spy_apply(engine, *args, **kwargs):
            # Observe the real adapter installed by _supervised_anchor_cycle for
            # THIS cycle, then delegate to the genuine implementation.
            observed[current["cond"]] = isinstance(engine.adapter, BoundedAdapter)
            return original_apply(engine, *args, **kwargs)

        with self._patched_swap_backend(), patch.object(
            mod, "_apply_adapter_to_all_docs", side_effect=spy_apply
        ):
            for cond in ("C3a", "C4a"):
                current["cond"] = cond
                run_experiment(
                    self._supervised_swap_config(cond, Path(self.tempdir.name) / f"{cond}-livebnd.json"),
                    corpus,
                    queries,
                    qrels,
                )
        self.assertTrue(observed["C3a"], "C3a's live per-cycle adapter must be bounded")
        self.assertFalse(observed["C4a"], "C4a's live per-cycle adapter must be unbounded")

    def test_c3a_requires_query_encoder_swap_and_anchor_fraction(self):
        with self._patched_swap_backend():
            # Wrong drift mode.
            bad_drift = DriftRecoveryConfig(
                **{**self._config("C3a", Path(self.tempdir.name) / "bad-c3a-drift.json").__dict__,
                   "anchor_fraction": 0.5}
            )
            with self.assertRaisesRegex(ValueError, "query_encoder_swap"):
                run_experiment(bad_drift, self.corpus, self.queries, self.qrels)
            # Right drift, but anchor_fraction == 0 (no anchors to supervise).
            bad_anchor = self._swap_config("C3a", Path(self.tempdir.name) / "bad-c3a-anchor.json")
            with self.assertRaisesRegex(ValueError, "anchor_fraction"):
                run_experiment(bad_anchor, self.corpus, self.queries, self.qrels)
            # C4a has the same requirement.
            bad_c4a = self._swap_config("C4a", Path(self.tempdir.name) / "bad-c4a-anchor.json")
            with self.assertRaisesRegex(ValueError, "anchor_fraction"):
                run_experiment(bad_c4a, self.corpus, self.queries, self.qrels)

    def test_anchor_fraction_zero_preserves_arena_behavior(self):
        """With anchor_fraction == 0 (default), C0/C2/C2O measure on ALL queries
        exactly as the A2a arena did — the split is inactive and the existing
        oracle-breaker numbers are unchanged.
        """
        corpus, queries, qrels = self._swap_corpus_queries_qrels()
        with self._patched_swap_backend():
            result = run_experiment(
                self._swap_config("C0", Path(self.tempdir.name) / "af0-c0.json"),
                corpus,
                queries,
                qrels,
            )
        split = result["anchor_eval_split"]
        self.assertFalse(split["active"])
        self.assertEqual(split["anchor_count"], 0)
        self.assertEqual(split["eval_count"], len(queries))

    def _patched_backend(self):
        @contextmanager
        def manager():
            with ExitStack() as stack:
                stack.enter_context(
                    patch("antigravity_engine.create_embedding_backend", return_value=TinyEmbeddingBackend())
                )
                stack.enter_context(patch("antigravity_engine.get_logger", return_value=MagicMock()))
                yield

        return manager()

    def _pre_change_c3_default_golden_trajectory(self):
        base_metadata = {
            "action": "detection_triggered_sedimentation",
            "annealing_observation": {
                "drift_magnitude": 0.2132925720177655,
                "should_correct": True,
                "temperature": 0.2132925720177655,
            },
            "annealing_settings": {
                "effective_epochs": 1,
                "effective_learning_rate": 0.00029196331481598893,
                "learning_rate_scale": 0.2919633148159889,
                "online_intensity": 0.2132925720177655,
                "original_epochs": 1,
                "original_learning_rate": 0.001,
                "temperature": 0.2132925720177655,
            },
            "bounded": True,
            "condition": "C3",
            "correction_applied": False,
            "correction_norm_stats": {
                "count": 4,
                "max": 0.009997744113206863,
                "mean": 0.007490877062082291,
                "sample_norms": [
                    0.009968805126845837,
                    0.0,
                    0.009997744113206863,
                    0.009996959008276463,
                ],
            },
            "detector_source": "AntigravityEngine._compute_annealing_drift_magnitude",
            "evaluated_queries": 2,
            "query_ndcg": [
                {
                    "ndcg": 1.0,
                    "query_id": "q-alpha",
                    "ranked_ids": ["d-alpha", "d-delta", "d-gamma", "d-beta"],
                    "relevant_ids": ["d-alpha"],
                },
                {
                    "ndcg": 1.0,
                    "query_id": "q-beta",
                    "ranked_ids": ["d-beta", "d-gamma", "d-delta", "d-alpha"],
                    "relevant_ids": ["d-beta"],
                },
            ],
            "sedimentation_attempted": True,
            "should_correct": True,
        }
        return [
            {"cycle_index": 1, "metadata": base_metadata, "ndcg": 1.0},
            {"cycle_index": 2, "metadata": dict(base_metadata), "ndcg": 1.0},
        ]


if __name__ == "__main__":
    unittest.main()
