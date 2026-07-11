import os
import unittest
from inspect import isclass

from tests.conftest import ensure_dummy_modules


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestInfluenceMethodRegistration(unittest.TestCase):
    def test_registration_sgd(self):
        ensure_dummy_modules()
        # Import the package to trigger decorator registration (sgd is served
        # by the canonical adapter implementation).
        import wie.infl  # noqa: F401
        from wie.infl.core import InfluenceCalculatorFactory, InfluenceCalculator

        self.assertIn("sgd", InfluenceCalculatorFactory._calculators)
        cls = InfluenceCalculatorFactory._calculators["sgd"]
        self.assertTrue(isclass(cls))
        self.assertTrue(issubclass(cls, InfluenceCalculator))

    def test_registration_icml(self):
        ensure_dummy_modules()
        import wie.infl  # noqa: F401
        from wie.infl.core import InfluenceCalculatorFactory, InfluenceCalculator

        self.assertIn("icml", InfluenceCalculatorFactory._calculators)
        cls = InfluenceCalculatorFactory._calculators["icml"]
        self.assertTrue(isclass(cls))
        self.assertTrue(issubclass(cls, InfluenceCalculator))

    def test_registration_wie_all_epochs(self):
        ensure_dummy_modules()
        from wie.infl.core import (
            InfluenceCalculatorFactory,
            InfluenceCalculator,
        )
        from wie.models.networks import get_network
        from wie.infl.vis import sum_norm

        # Clear existing registration if any
        if "wie_all_epochs" in InfluenceCalculatorFactory._calculators:
            del InfluenceCalculatorFactory._calculators["wie_all_epochs"]

        # Read the source file by path (cwd-independent) and exec it, pinning
        # both the file location and the class name (Phase 2 updates both).
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

        self.assertIn("wie_all_epochs", InfluenceCalculatorFactory._calculators)
        cls = InfluenceCalculatorFactory._calculators["wie_all_epochs"]
        self.assertTrue(isclass(cls))
        self.assertTrue(issubclass(cls, InfluenceCalculator))

    def test_registration_loo_true(self):
        """
        The LOO method in this codebase is exposed as the 'true' influence
        calculation (counterfactual leave-one-out). Ensure it is registered.
        """
        ensure_dummy_modules()
        import wie.infl.true_influence  # noqa: F401
        from wie.infl.core import InfluenceCalculatorFactory, InfluenceCalculator

        self.assertIn("true", InfluenceCalculatorFactory._calculators)
        cls = InfluenceCalculatorFactory._calculators["true"]
        self.assertTrue(isclass(cls))
        self.assertTrue(issubclass(cls, InfluenceCalculator))

    def test_registration_lava(self):
        ensure_dummy_modules()
        import wie.infl.lava  # noqa: F401
        from wie.infl.core import InfluenceCalculatorFactory, InfluenceCalculator

        self.assertIn("lava", InfluenceCalculatorFactory._calculators)
        cls = InfluenceCalculatorFactory._calculators["lava"]
        self.assertTrue(isclass(cls))
        self.assertTrue(issubclass(cls, InfluenceCalculator))


if __name__ == "__main__":
    unittest.main()
