"""CPU tests for Bundle 52 QSCCI and Bundle 53 PM-6 shadow probes."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pm6_shadow_probe import (  # noqa: E402
    ShadowTapEnvelope,
    _hidden_from_text,
    execute as execute_shadow,
    main as main_shadow,
    pm6_release_bytes,
    run_shadow_drycheck,
)
from chelated_pv7_spine.burn_ledger import text_fingerprint  # noqa: E402
from chelated_pv7_spine.fragment_compiler import compile_repo_fragments  # noqa: E402
from chelated_pv7_spine.claim_registry import HOLM_FAMILY, claim_registry_sha256  # noqa: E402
from chelated_qscci_singleton_probe import (  # noqa: E402
    ARM_NAMES,
    DOSE_NORM,
    execute as execute_qscci,
    main as main_qscci,
    run_qscci_experiment,
    sae_decoder_direction,
)
from scripts.run_pv7_w18_3c_qscci_singleton import (  # noqa: E402
    execute as runner_qscci,
    main as runner_qscci_main,
)
from scripts.run_pv7_w18_3d_pm6_shadow import (  # noqa: E402
    execute as runner_shadow,
    main as runner_shadow_main,
)

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"


class QscciProbeTests(unittest.TestCase):
    def test_positive_control_and_matched_arms(self) -> None:
        opened = {"report": False}

        def on_report() -> None:
            opened["report"] = True

        report = run_qscci_experiment(readout="causal", on_report=on_report)
        self.assertTrue(opened["report"])
        self.assertTrue(report["positive_control"]["passed"])
        self.assertGreater(report["positive_control"]["cosine_qscci_w_true"], 0.99)
        self.assertGreater(
            report["positive_control"]["cosine_correlation_w_nuis"],
            report["positive_control"]["cosine_correlation_w_true"],
        )
        self.assertTrue(report["report_emitted"])
        self.assertEqual([arm["name"] for arm in report["arms"]], list(ARM_NAMES))
        self.assertTrue(all(abs(arm["delta_h_l2"] - DOSE_NORM) < 1e-8 for arm in report["arms"]))
        self.assertLess(report["norm_spread"], 1e-8)
        self.assertLess(report["selection_cosine_qscci_caa"], 0.95)
        self.assertTrue(report["nuisance_invariance_win"])
        self.assertLess(report["restoration_max_abs_error"], 1e-4)
        self.assertGreater(report["restoration_corrupt_baseline_mae"], 0.1)
        self.assertGreater(report["restoration_fraction"], 0.99)
        self.assertGreater(report["sham_restoration_max_abs_error"], 0.1)
        self.assertLess(report["sham_restoration_fraction"], 0.25)
        self.assertTrue(report["native_beats_sham"])
        self.assertEqual(report["burn_inventory_scope"], "local_drycheck_sentinel")
        native = report["arms"][0]
        self.assertGreater(native["removal_mean_abs_delta_y"], 0.0)
        self.assertTrue(native["restoration_two_pass_patched"])
        for arm in report["arms"]:
            self.assertEqual(arm["sampling_unit"], "family")
            self.assertEqual(arm["n_families"], 32)
            self.assertEqual(arm["bootstrap_resamples"], 2000)
        collateral = report["arms"][4]
        self.assertGreater(collateral["native_intervention_collateral_retention"], 0.99)

        # Deterministic SAE direction & scoreboard hash across repeated runs.
        second = execute_qscci("local-drycheck")
        first = execute_qscci("local-drycheck")
        self.assertEqual(first["scoreboard_sha256"], second["scoreboard_sha256"])
        import numpy as np

        sample = np.random.default_rng(7).normal(size=(16, 32))
        self.assertEqual(sae_decoder_direction(sample).tolist(), sae_decoder_direction(sample).tolist())

    def test_fail_closed_before_report(self) -> None:
        opened = {"report": False}

        def on_report() -> None:
            opened["report"] = True

        report = run_qscci_experiment(readout="nuisance", on_report=on_report)
        self.assertFalse(report["positive_control"]["passed"])
        self.assertFalse(report["report_emitted"])
        self.assertEqual(report["arms"], [])
        self.assertFalse(opened["report"])
        self.assertEqual(report["failure"], "positive_control")

    def test_live_gpu_refuses_weight_load(self) -> None:
        report = runner_qscci("run-live-gpu", "Qwen/Qwen3.5-2B-Base")
        self.assertFalse(report["executed"])
        self.assertFalse(report["passed"])
        self.assertFalse(report["report_emitted"])
        self.assertFalse(report["live_weights_loaded"])
        self.assertFalse(report["qwen_report_split_executed"])
        self.assertTrue(report["requires_real_weight_file_sha256"])
        self.assertTrue(report["requires_historical_burn_inventory"])
        self.assertEqual(report["reason"], "cuda-isolated-live-weights-not-loaded")
        self.assertEqual(main_qscci(["--mode", "run-live-gpu"]), 2)
        self.assertEqual(runner_qscci_main(["--mode", "run-live-gpu"]), 2)
        unknown = execute_qscci("run-live-gpu", "not-a-preregistered-model")
        self.assertFalse(unknown["passed"])
        self.assertEqual(unknown["reason"], "model-not-in-preregistered-set")


class ShadowProbeTests(unittest.TestCase):
    def test_mutation_and_throw_leave_release_bytes_identical(self) -> None:
        text = "alpha beta gamma delta epsilon zeta"
        hidden = _hidden_from_text(text)
        decision = {"action": "release", "nested": {"keep": 1}}
        envelope = ShadowTapEnvelope.capture(hidden, text.encode("utf-8"), decision)

        def mutate(env: ShadowTapEnvelope) -> bytes:
            env.hidden += 50.0
            env.stream.extend(b"TAMPER")
            env.decision["action"] = "abort"
            env.decision["nested"]["keep"] = 0
            return b"TAMPER"

        def explode(env: ShadowTapEnvelope) -> None:
            env.hidden += 1.0
            raise RuntimeError("probe failed")

        from chelated_pm6_shadow_probe import shadow_gate_return

        self.assertIsNone(shadow_gate_return(envelope, mutate))
        self.assertEqual(hidden.tolist(), _hidden_from_text(text).tolist())
        self.assertEqual(decision["action"], "release")
        self.assertEqual(decision["nested"]["keep"], 1)
        alone = pm6_release_bytes(text, probe=None)
        self.assertGreater(len(alone), 0)
        self.assertEqual(alone, pm6_release_bytes(text, probe=mutate))
        self.assertEqual(alone, pm6_release_bytes(text, probe=explode))

    def test_drycheck_bound_and_observational_lead_time(self) -> None:
        report = run_shadow_drycheck()
        self.assertGreaterEqual(report["report_family_count"], 320)
        self.assertEqual(report["byte_identical_families"], 320)
        for record in report["incumbent_pm6_plus_shadow"].values():
            self.assertLess(record["holm_adjusted_cp95_upper"], 0.02)
            self.assertLess(record["holm_adjusted_p_value"], 0.05)
            self.assertEqual(record["false_alerts"], 0)
            self.assertGreater(record["report_max_score"], 0.0)
            self.assertGreaterEqual(record["threshold"], 4.5)
        self.assertEqual(report["burn_inventory_scope"], "local_drycheck_sentinel")
        holm = report["holm_family_disclosure"]
        self.assertFalse(holm["included_in_t1_confirmatory_holm_family"])
        self.assertEqual(tuple(holm["t1_claim_registry_holm_family"]), HOLM_FAMILY)
        telemetry = report["telemetry"]
        self.assertFalse(telemetry["counted_in_false_alert_denominator"])
        self.assertFalse(telemetry["r3_confirmatory"])
        self.assertEqual(telemetry["r3_tier"], "T3")
        self.assertEqual(telemetry["r3_lead_time_status"], "not_measured_in_local_drycheck")
        self.assertIsNone(telemetry["r3_lead_min"])
        self.assertEqual(telemetry["r3_lead_times"], [])
        self.assertEqual(telemetry["singleton_hits"], telemetry["sensitivity_n"])
        self.assertEqual(telemetry["sparse_hits"], telemetry["sensitivity_n"])
        seen = set()
        for rows in report["family_ids"].values():
            self.assertFalse(seen.intersection(rows))
            seen.update(rows)

    def test_live_shadow_refuses_protected_name_without_opening(self) -> None:
        refused = runner_shadow("run-live-shadow", str(REPO_ROOT / "not-created" / "sealed_key.json"))
        self.assertFalse(refused["executed"])
        self.assertFalse(refused["passed"])
        self.assertFalse(refused["opened"])
        self.assertEqual(refused["reason"], "protected-path-refused")
        traversal = runner_shadow("run-live-shadow", str(REPO_ROOT / "scripts" / ".." / "sealed_key.json"))
        self.assertFalse(traversal["executed"])
        self.assertFalse(traversal["passed"])
        self.assertEqual(traversal["reason"], "protected-path-refused")
        missing = execute_shadow("run-live-shadow", str(REPO_ROOT / "not-created" / "ordinary-manifest.json"))
        self.assertFalse(missing["executed"])
        self.assertFalse(missing["passed"])
        self.assertEqual(missing["reason"], "manifest-absent")
        self.assertEqual(main_shadow(["--mode", "run-live-shadow"]), 2)
        self.assertEqual(runner_shadow_main(["--mode", "run-live-shadow"]), 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shadow-manifest.json"
            path.write_text(json.dumps({"prompts": ["alpha beta gamma delta epsilon zeta"]}), encoding="utf-8")
            # Without historical burn inventory, run-live-shadow must fail closed (rc=2).
            no_burn = execute_shadow("run-live-shadow", str(path))
            self.assertFalse(no_burn["executed"])
            self.assertFalse(no_burn["passed"])
            self.assertEqual(no_burn["reason"], "historical-burn-inventory-required")
            self.assertEqual(len(no_burn["manifest_sha256"]), 64)
            self.assertEqual(runner_shadow_main(["--mode", "run-live-shadow", "--pm6-manifest", str(path)]), 2)

            # With real historical burn inventory, observational manifest executes cleanly.
            hist = {
                "documents": [text_fingerprint("historical doc shadow test")],
                "queries": [text_fingerprint("historical query shadow test")],
            }
            observed = execute_shadow(
                "run-live-shadow", str(path), historical_fingerprints=hist, allow_local_drycheck_smoke=True
            )
        self.assertTrue(observed["executed"])
        self.assertTrue(observed["passed"])
        self.assertFalse(observed["report_emitted"])
        self.assertFalse(observed["live_pm6_gateway_executed"])
        self.assertEqual(len(observed["manifest_sha256"]), 64)
        self.assertEqual(observed["burn_inventory_scope"], "supplied_historical_inventory")
        self.assertEqual(observed["byte_identical_families"], 1)


class FragmentCompilerTests(unittest.TestCase):
    def test_repo_fragments_include_both_verifiers(self) -> None:
        compiled = compile_repo_fragments(REPO_ROOT, claim_registry_sha256())
        self.assertEqual(claim_registry_sha256(), SPINE_DIGEST)
        ids = [row["id"] for row in compiled["registry"]]
        self.assertIn(52, ids)
        self.assertIn(53, ids)
        verifier_ids = [row["bundle_id"] for row in compiled["verifiers"]]
        self.assertIn(52, verifier_ids)
        self.assertIn(53, verifier_ids)
        order = compiled["dependency_order"]
        self.assertLess(order.index(52), len(order))
        self.assertLess(order.index(53), len(order))


if __name__ == "__main__":
    unittest.main()
