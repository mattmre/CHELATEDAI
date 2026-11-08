from __future__ import annotations

import unittest
import warnings
from pathlib import Path

from setuptools import find_packages
from setuptools.config.pyprojecttoml import read_configuration

import egv.recommissioning as recommissioning


class EgvPackageManifestTests(unittest.TestCase):
    def test_setuptools_manifest_includes_every_egv_package(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        discovered = set(find_packages(where=str(repository_root), include=("egv", "egv.*")))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            configuration = read_configuration(str(repository_root / "pyproject.toml"))
        configured = set(configuration["tool"]["setuptools"]["packages"])

        self.assertEqual(configured, discovered)
        self.assertIn("egv.recommissioning", configured)

    def test_recommissioning_public_exports_are_bound(self) -> None:
        self.assertTrue(recommissioning.__all__)
        for name in recommissioning.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(recommissioning, name))


if __name__ == "__main__":
    unittest.main()
