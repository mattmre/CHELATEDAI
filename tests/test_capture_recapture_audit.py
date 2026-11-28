"""Unit tests for capture_recapture_audit (W22-08). unittest only."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import capture_recapture_audit as cra  # noqa: E402


class TestPairwise(unittest.TestCase):
    def test_lincoln_petersen(self):
        self.assertAlmostEqual(cra.lincoln_petersen(10, 12, 4), 30.0)
        self.assertTrue(math.isinf(cra.lincoln_petersen(3, 3, 0)))

    def test_chapman_hand_values(self):
        c = cra.chapman(10, 12, 4)
        # (11*13)/5 - 1 = 27.6 ; var = 11*13*6*8 / (25*6) = 45.76
        self.assertAlmostEqual(c["N"], 27.6)
        self.assertAlmostEqual(c["var"], 45.76)
        self.assertAlmostEqual(c["ci95"][1] - c["N"], cra.Z95 * math.sqrt(45.76))

    def test_overlap_and_jaccard(self):
        p = cra.pairwise({"a": {"x", "y", "z", "w"}, "b": {"x", "y"}})[0]
        self.assertEqual(p["m"], 2)
        self.assertAlmostEqual(p["overlap"], 1.0)
        self.assertAlmostEqual(p["jaccard"], 0.5)


class TestChao2(unittest.TestCase):
    def test_f2_positive(self):
        cap = {"a": {1, 2, 3, 4}, "b": {1, 2, 5}, "c": {1, 6}}
        r = cra.chao2(cap)
        self.assertEqual((r["S_obs"], r["f1"], r["f2"]), (6, 4, 1))
        a = 2 / 3
        self.assertAlmostEqual(r["f0"], a * 16 / 2)  # 5.333
        var = 1 * (a / 2 * 16 + a * a * 64 + a * a / 4 * 256)
        self.assertAlmostEqual(r["var_f0"], var)
        c = math.exp(cra.Z95 * math.sqrt(math.log(1 + var / r["f0"] ** 2)))
        self.assertAlmostEqual(r["f0_upper95"], r["f0"] * c)
        self.assertAlmostEqual(r["N_ci95"][1], 6 + r["f0"] * c)

    def test_f2_zero_bias_corrected(self):
        r = cra.chao2({"a": {1}, "b": {2}, "c": {3}})
        self.assertEqual((r["f1"], r["f2"]), (3, 0))
        self.assertAlmostEqual(r["f0"], (2 / 3) * 3 * 2 / 2)

    def test_f0_zero(self):
        r = cra.chao2({"a": {1, 2}, "b": {1, 2}, "c": {1, 2}})
        self.assertEqual(r["f0"], 0)
        self.assertEqual(r["f0_upper95"], 0.0)
        self.assertTrue(r["degenerate_f0_zero"])


class TestLogLinear(unittest.TestCase):
    def test_two_list_independence_equals_lp_residual(self):
        # histories (0,1), (1,0), (1,1): saturated 3-param fit, f0 = n10 * n01 / n11
        x = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)
        y = np.array([6.0, 8.0, 4.0])
        fit = cra.fit_poisson_irls(x, y)
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(math.exp(fit["beta"][0]), 8 * 6 / 4, places=6)

    def test_three_list_exact_independence(self):
        # expected counts of N=1000 with capture probs 0.5/0.4/0.3: all-zero cell = 1000*.5*.6*.7 = 210
        p = (0.5, 0.4, 0.3)
        ids, n = {"a": set(), "b": set(), "c": set()}, 0
        for h in [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
            cnt = round(1000 * np.prod([p[i] if h[i] else 1 - p[i] for i in range(3)]))
            for _ in range(int(cnt)):
                n += 1
                for i, lid in enumerate("abc"):
                    if h[i]:
                        ids[lid].add(n)
        models = cra.loglinear_models(ids)
        self.assertEqual(models[0]["model"], "independence")
        self.assertAlmostEqual(models[0]["f0"], 210.0, places=4)
        self.assertEqual(len(models), 4)
        self.assertFalse(any(m["boundary_estimate"] for m in models))

    def test_subset_list_flags_boundary(self):
        # b and c are subsets of a: cells with a=0 are empty, MLE does not exist
        cap = {"a": set(range(10)), "b": {0, 1, 2, 3}, "c": {2, 3, 4}}
        models = cra.loglinear_models(cap)
        self.assertTrue(models[0]["boundary_estimate"])
        k3 = cra.k3(models, 10)
        self.assertFalse(k3["pass"])


class TestKillRules(unittest.TestCase):
    def test_k1_k2(self):
        self.assertFalse(cra.k1(2)["pass"])
        self.assertTrue(cra.k1(3)["pass"])
        pairs = cra.pairwise({"a": set(range(10)), "b": set(range(9)), "c": {20, 21}})
        r = cra.k2(pairs)
        self.assertFalse(r["pass"])
        self.assertEqual(r["failing_pairs"], [["a", "b"]])

    def test_k3_ratio_and_theta(self):
        models = [{"model": "independence", "f0": 4.0, "finite": True, "boundary_estimate": False},
                  {"model": "a*b", "f0": 13.0, "finite": True, "boundary_estimate": False}]
        r = cra.k3(models, 20)
        self.assertAlmostEqual(r["f0_ratio"], 13 / 4)
        self.assertTrue(r["a_fail"])
        self.assertFalse(r["b_fail"])
        self.assertEqual(r["f0_theta"]["4.0"], 16.0)
        r2 = cra.k3(models[:1], 7)  # 2*4 = 8 > 7
        self.assertTrue(r2["b_fail"])

    def test_k4_unique_counts(self):
        cap = {"a": {1, 2, 3, 9}, "b": {1, 2, 4}, "c": {1, 3, 5}}
        r = cra.k4(cap)
        u = {row["held_out"]: row["u_i"] for row in r["rows"]}
        self.assertEqual(u, {"a": 1, "b": 1, "c": 1})
        # held-out a: others b,c -> f1=4 (2,4,3,5), f2=1 (1); k=2 -> f0 = 0.5*16/2 = 4
        self.assertAlmostEqual(r["rows"][0]["chao2_others_f0"], 4.0)


class TestFiltersAndVerification(unittest.TestCase):
    def test_verify_findings(self):
        src = {"L": ["first line of memo", "the verifier does not open the live receipt at all"]}
        fs = [
            {"finding_id": "ok", "list_id": "L", "source_line": 2, "quote": "does not open the live receipt"},
            {"finding_id": "short", "list_id": "L", "source_line": 2, "quote": "does not open"},
            {"finding_id": "wrongline", "list_id": "L", "source_line": 1, "quote": "does not open the live receipt"},
            {"finding_id": "range", "list_id": "L", "source_line": 9, "quote": "does not open the live receipt"},
        ]
        ok, bad = cra.verify_findings(fs, src)
        self.assertEqual([f["finding_id"] for f in ok], ["ok"])
        self.assertEqual(len(bad), 3)

    def test_build_unit_scope_independence_and_as_if(self):
        fam = {"lists": [
            {"list_id": "g", "declared_scope": ["B1", "B2", "XC"]},
            {"list_id": "s", "declared_scope": ["B3"]},
            {"list_id": "j1", "declared_scope": ["B1", "B2", "B3"], "reads_other_lists": True},
            {"list_id": "j2", "declared_scope": ["B1", "B2", "B3"], "reads_other_lists": True},
        ]}
        fs = [
            {"finding_id": "g-1", "list_id": "g", "ordinals": ["B1", "XC"], "defect_class": "C01"},
            {"finding_id": "g-2", "list_id": "g", "ordinals": ["B2"], "defect_class": "C03"},
            {"finding_id": "g-3", "list_id": "g", "ordinals": ["B1"], "defect_class": "C01"},
            {"finding_id": "j1-1", "list_id": "j1", "ordinals": ["B1"], "defect_class": "C01"},
            {"finding_id": "j2-1", "list_id": "j2", "ordinals": ["B2"], "defect_class": "C03"},
        ]
        u = cra.build_unit(fam, fs, {"B1", "B2"})
        self.assertEqual(u["lists_in_scope"], ["g", "j1", "j2"])
        self.assertEqual(u["lists_dependent"], ["j1", "j2"])
        self.assertEqual(u["estimating_set"], ["g"])
        self.assertFalse(u["gating"]["K1"]["pass"])
        self.assertFalse(u["gating"]["passes"])
        self.assertIn("as_if_independent_non_gating", u)
        self.assertEqual(u["captures_all_lists"]["g"], ["B1:C01", "B2:C03"])  # duplicate collapsed, XC dropped
        self.assertEqual(len(u["dropped_out_of_scope"]), 1)

    def test_brief_preseed_marks_all_dependent(self):
        fam = {"brief_preseeds_defects": True,
               "lists": [{"list_id": x, "declared_scope": ["B1"]} for x in "abc"]}
        u = cra.build_unit(fam, [], {"B1"})
        self.assertEqual(u["estimating_set"], [])


if __name__ == "__main__":
    unittest.main()
