import unittest

from computational_storage_poc.disk_llm_estimator import estimate_disk_llm


class DiskLLMEstimatorTests(unittest.TestCase):
    def test_weight_size_matches_simple_quantization_math(self):
        estimate = estimate_disk_llm(
            params_billion=70,
            bits_per_weight=4,
            ssd_bandwidth_gbps=14,
            dram_gb=128,
        )
        self.assertAlmostEqual(estimate.model_size_gb, 35.0, places=6)
        self.assertAlmostEqual(estimate.resident_set_gb, 35.0 / 3.0, places=6)

    def test_sparse_flash_path_beats_dense_streaming_upper_bound(self):
        estimate = estimate_disk_llm(
            params_billion=70,
            bits_per_weight=4,
            ssd_bandwidth_gbps=14,
            dram_gb=128,
        )
        self.assertGreater(estimate.flash_tokens_per_second_upper, estimate.dense_tokens_per_second_upper)

    def test_ram_and_paper_ratio_flags_can_fail_independently(self):
        estimate = estimate_disk_llm(
            params_billion=405,
            bits_per_weight=4,
            ssd_bandwidth_gbps=7,
            dram_gb=64,
        )
        self.assertFalse(estimate.fits_resident_ram)
        self.assertFalse(estimate.fits_paper_ratio)


if __name__ == "__main__":
    unittest.main()
