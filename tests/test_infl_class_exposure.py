import os
import unittest

from tests.conftest import ensure_dummy_modules


# Repository root, resolved from this file so the file-path test below works
# regardless of the current working directory (structural gate runs from /tmp).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestInfluenceClassExposure(unittest.TestCase):
    def test_sgd_class_exposed(self):
        ensure_dummy_modules()
        # sgd is served by the canonical, deduplicated adapter implementation.
        from wie.infl.adapters import SgdAdapterCalculator
        from wie.infl.core import InfluenceCalculator

        self.assertTrue(issubclass(SgdAdapterCalculator, InfluenceCalculator))

    def test_icml_class_exposed(self):
        ensure_dummy_modules()
        from wie.infl.adapters import IcmlAdapterCalculator
        from wie.infl.core import InfluenceCalculator

        self.assertTrue(issubclass(IcmlAdapterCalculator, InfluenceCalculator))

    def test_wie_all_epochs_class_exposed(self):
        ensure_dummy_modules()
        from wie.infl.core import InfluenceCalculator
        from wie.models.networks import get_network
        from wie.infl.vis import sum_norm

        # Read the source file by path and exec it, pinning both the file
        # location and the class name (Phase 2 rename updates both).
        src_path = os.path.join(REPO_ROOT, "src", "wie", "infl", "wie_all_epochs.py")
        with open(src_path, "r") as f:
            source_code = f.read()

        modified_source = source_code.replace(
            "from .core import (", "from wie.infl.core import ("
        )
        test_namespace = {
            "__name__": "test_wie_all_epochs",
            "get_network": get_network,
            "sum_norm": sum_norm,
        }
        exec(modified_source, test_namespace)

        WieAllEpochsInfluenceCalculator = test_namespace[
            "WieAllEpochsInfluenceCalculator"
        ]
        self.assertTrue(
            issubclass(WieAllEpochsInfluenceCalculator, InfluenceCalculator)
        )

    def test_true_loo_class_exposed(self):
        """LOO method is implemented as 'true' influence in this repository."""
        ensure_dummy_modules()
        from wie.infl.true_influence import TrueInfluenceCalculator
        from wie.infl.core import InfluenceCalculator

        self.assertTrue(issubclass(TrueInfluenceCalculator, InfluenceCalculator))

    def test_lava_class_exposed(self):
        ensure_dummy_modules()
        from wie.infl.lava import LavaInfluenceCalculator
        from wie.infl.core import InfluenceCalculator

        self.assertTrue(issubclass(LavaInfluenceCalculator, InfluenceCalculator))


if __name__ == "__main__":
    unittest.main()
