from __future__ import annotations

import copy
import unittest

from research.drift_recovery.d2.decisions import (
    FROZEN_CONTRASTS,
    apply_kill_screen,
    evaluate_g2,
)


def _passing_contrast() -> dict:
    return {
        "delta_ndcg": {
            "estimate": 0.02,
            "ci_low": 0.001,
            "ci_high": 0.04,
            "half_width": 0.0195,
        },
        "recovery_point_advantage": {
            "estimate": 0.05,
            "ci_low": 0.001,
            "ci_high": 0.09,
            "half_width": 0.0445,
        },
        "seed_delta_ndcg": [0.02, 0.021, 0.019, 0.023, 0.018],
        "multiple_testing": {"reject": True, "holm_adjusted_p": 0.01},
        "half_width": 0.01,
    }


def _passing_family() -> dict:
    return {name: _passing_contrast() for name in FROZEN_CONTRASTS}


class TestG2PowerGate(unittest.TestCase):
    def test_g2_uses_median_primary_half_width(self) -> None:
        widths = dict(zip(FROZEN_CONTRASTS, (0.010, 0.020, 0.012)))
        contrasts = {name: {"half_width": width} for name, width in widths.items()}

        result = evaluate_g2(contrasts, query_count=100, threshold=0.015)

        self.assertAlmostEqual(result["median_half_width"], 0.012)
        self.assertTrue(result["pass"])
        self.assertEqual(result["family"], list(FROZEN_CONTRASTS))

    def test_g2_failure_hard_short_circuits_g3(self) -> None:
        widths = dict(zip(FROZEN_CONTRASTS, (0.020, 0.030, 0.010)))
        contrasts = {name: {"half_width": width} for name, width in widths.items()}
        g2 = evaluate_g2(contrasts, query_count=100, threshold=0.015)
        self.assertFalse(g2["pass"])
        self.assertAlmostEqual(g2["median_half_width"], 0.020)

        # An empty G3 contrast mapping proves the function returns before trying
        # to interpret any correction result.
        decision = apply_kill_screen(
            g2=g2,
            contrasts={},
            detector_auprc=1.0,
            min_oracle_gap=1.0,
        )
        self.assertEqual(decision["status"], "UNDERPOWERED_NEGATIVE")
        self.assertFalse(decision["g3_interpreted"])
        self.assertFalse(decision["win"])
        self.assertEqual(decision["verdict"], "NO_G3_VERDICT")

    def test_invalid_oracle_gap_draw_fraction_blocks_g3_after_g2(self) -> None:
        decision = apply_kill_screen(
            g2={"pass": True},
            contrasts={},
            detector_auprc=1.0,
            min_oracle_gap=1.0,
            invalid_oracle_gap_fraction=0.0101,
        )
        self.assertEqual(decision["status"], "INVALID_ORACLE_GAP_BOOTSTRAP")
        self.assertFalse(decision["g3_interpreted"])
        self.assertEqual(decision["verdict"], "NO_G3_VERDICT")

    def test_identity_primary_arm_blocks_g3_after_power_and_gap_checks(self) -> None:
        decision = apply_kill_screen(
            g2={"pass": True},
            contrasts={},
            detector_auprc=1.0,
            min_oracle_gap=1.0,
            invalid_oracle_gap_fraction=0.0,
            primary_movement_valid=False,
        )
        self.assertEqual(decision["status"], "INVALID_IDENTITY_PRIMARY_ARM")
        self.assertFalse(decision["g3_interpreted"])


class TestG3DualCIDecision(unittest.TestCase):
    @staticmethod
    def _apply(contrasts=None, auprc=0.80, oracle_gap=0.05):
        return apply_kill_screen(
            g2={"pass": True},
            contrasts=contrasts if contrasts is not None else _passing_family(),
            detector_auprc=auprc,
            min_oracle_gap=oracle_gap,
            seed_count=5,
        )

    def test_powered_boundary_case_requires_every_dual_ci_axis(self) -> None:
        decision = self._apply()

        self.assertTrue(decision["g3_interpreted"])
        self.assertTrue(decision["detection"]["pass"])
        self.assertTrue(decision["correction"]["oracle_gap_pass"])
        self.assertTrue(decision["correction"]["pass"])
        self.assertTrue(decision["win"])
        self.assertEqual(decision["verdict"], "SURPRISING_CHELATION_WIN")
        for contrast in decision["correction"]["contrasts"].values():
            self.assertTrue(contrast["pass"])
            self.assertTrue(all(contrast["checks"].values()))

    def test_each_correction_axis_can_kill_the_win(self) -> None:
        failures = {
            "ndcg_estimate": ("delta_ndcg", "estimate", 0.0199),
            "ndcg_ci": ("delta_ndcg", "ci_low", 0.0),
            "recovery_estimate": ("recovery_point_advantage", "estimate", 0.0499),
            "recovery_ci": ("recovery_point_advantage", "ci_low", 0.0),
            "holm": ("multiple_testing", "reject", False),
        }
        target = FROZEN_CONTRASTS[0]
        for label, (section, field, value) in failures.items():
            with self.subTest(axis=label):
                family = _passing_family()
                family[target][section][field] = value
                decision = self._apply(contrasts=family)
                self.assertFalse(decision["correction"]["contrasts"][target]["pass"])
                self.assertFalse(decision["correction"]["pass"])
                self.assertFalse(decision["win"])
                self.assertEqual(
                    decision["verdict"], "KILL_CORRECTOR_PARK_AS_REEMBED_ROUTER"
                )

    def test_seed_sign_requires_exactly_five_strictly_positive_seeds(self) -> None:
        target = FROZEN_CONTRASTS[1]
        for label, seed_values in (
            ("one_negative", [0.02, 0.01, -0.001, 0.02, 0.03]),
            ("one_zero", [0.02, 0.01, 0.0, 0.02, 0.03]),
            ("only_four", [0.02, 0.01, 0.02, 0.03]),
        ):
            with self.subTest(case=label):
                family = _passing_family()
                family[target]["seed_delta_ndcg"] = seed_values
                decision = self._apply(contrasts=family)
                checks = decision["correction"]["contrasts"][target]["checks"]
                self.assertFalse(checks["same_positive_sign_all_seeds"])
                self.assertFalse(decision["win"])

    def test_oracle_gap_and_detector_thresholds_are_binding(self) -> None:
        low_gap = self._apply(oracle_gap=0.0499)
        self.assertFalse(low_gap["correction"]["oracle_gap_pass"])
        self.assertFalse(low_gap["win"])
        self.assertEqual(low_gap["verdict"], "KILL_CORRECTOR_PARK_AS_REEMBED_ROUTER")

        low_detection = self._apply(auprc=0.7999)
        self.assertFalse(low_detection["detection"]["pass"])
        self.assertTrue(low_detection["correction"]["pass"])
        self.assertFalse(low_detection["win"])
        self.assertEqual(low_detection["verdict"], "KILL_CORRECTOR")

        undefined_detection = self._apply(auprc=None)
        self.assertFalse(undefined_detection["detection"]["pass"])
        self.assertFalse(undefined_detection["win"])

    def test_failure_in_any_frozen_contrast_blocks_the_win(self) -> None:
        for target in FROZEN_CONTRASTS:
            with self.subTest(contrast=target):
                family = _passing_family()
                family[target]["delta_ndcg"]["estimate"] = 0.0
                self.assertFalse(self._apply(contrasts=family)["win"])

    def test_alpha_stress_and_local_diagnostic_are_not_decision_family(self) -> None:
        self.assertEqual(
            FROZEN_CONTRASTS,
            (
                "chelation_vs_cbie",
                "chelation_vs_hubness",
                "chelation_vs_global_ridge",
            ),
        )
        self.assertTrue(
            all("alpha" not in name and "local" not in name for name in FROZEN_CONTRASTS)
        )
        family = _passing_family()
        family["chelation_alpha_0_05_vs_cbie"] = copy.deepcopy(_passing_contrast())
        family["paired_local_adapter_vs_cbie"] = copy.deepcopy(_passing_contrast())

        decision = self._apply(contrasts=family)

        self.assertEqual(
            set(decision["correction"]["contrasts"]), set(FROZEN_CONTRASTS)
        )
        self.assertNotIn(
            "chelation_alpha_0_05_vs_cbie", decision["correction"]["contrasts"]
        )
        self.assertNotIn(
            "paired_local_adapter_vs_cbie", decision["correction"]["contrasts"]
        )


if __name__ == "__main__":
    unittest.main()
